"""
app/core/ocr/svtrv2_wrapper.py
Pure-PyTorch OpenOCR SVTRv2 Server text-line recognizer (`topdu/OpenOCR`, Apache-2.0).
Requires only PyTorch + numpy + OpenCV + huggingface_hub (zero PaddlePaddle/OpenOCR runtime bloat).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.ocr.transformers_rec_wrapper import (
    _release_torch_memory,
    expand_condensed_strip,
    recognize_box_lines,
    refine_condensed_line,
)

logger = logging.getLogger(__name__)

SVTRV2_REPO_ID = "topdu/OpenOCR"
SVTRV2_WEIGHTS_FILENAME = "openocr_svtrv2_ch.pth"
_DICT_PATH = Path(__file__).resolve().parent / "ppocr_keys_v1.txt"


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any = 3,
        stride: int = 1,
        padding: Any = 0,
        bias: bool = False,
        groups: int = 1,
        act: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer: type[nn.Module] = nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        act_layer: type[nn.Module] = nn.GELU,
        eps: float = 1e-6,
        prenorm: bool = True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.mixer = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            x = self.norm1(x + self.mixer(x))
            x = self.norm2(x + self.mlp(x))
        else:
            x = x + self.mixer(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class _ConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        eps: float = 1e-6,
        num_conv: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.mixer = nn.Sequential(
            *[nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=num_heads) for _ in range(num_conv)]
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape[1:]
        x = x + self.mixer(x)
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.mlp(x))
        return x.transpose(1, 2).reshape(-1, C, H, W)


class _FlattenTranspose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)


class _SubSample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=tuple(stride), padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, sz: List[int]) -> Tuple[torch.Tensor, List[int]]:
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class _SubSample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=tuple(stride), padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, sz: List[int]) -> Tuple[torch.Tensor, List[int]]:
        C = x.shape[-1]
        x = self.conv(x.transpose(1, 2).reshape(-1, C, sz[0], sz[1]))
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class _IdentitySize(nn.Module):
    def forward(self, x: torch.Tensor, sz: List[int]) -> Tuple[torch.Tensor, List[int]]:
        return x, sz


class _Feat2D(nn.Module):
    def forward(self, x: torch.Tensor, sz: List[int]) -> Tuple[torch.Tensor, List[int]]:
        C = x.shape[-1]
        return x.transpose(1, 2).reshape(-1, C, sz[0], sz[1]), sz


class _POPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, bias: bool = True):
        super().__init__()
        self.patch_embed = nn.Sequential(
            _ConvBNLayer(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=bias),
            _ConvBNLayer(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        H, W = x.shape[2:]
        return self.patch_embed(x), [H // 4, W // 4]


class _SVTRStage(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        mixer: Sequence[str],
        kernel_sizes: Sequence[int],
        sub_k: Sequence[int],
        num_heads: int,
        num_conv: Sequence[int],
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.Sequential()
        for i, mtype in enumerate(mixer):
            if mtype == "Conv":
                self.blocks.append(
                    _ConvBlock(
                        dim=dim,
                        num_heads=num_heads,
                        num_conv=num_conv[i],
                        kernel_size=kernel_sizes[i],
                    )
                )
            else:
                if mtype == "FGlobal":
                    self.blocks.append(_FlattenTranspose())
                self.blocks.append(_Block(dim=dim, num_heads=num_heads, prenorm=True))

        if downsample:
            if mixer[-1] == "Conv":
                self.downsample: nn.Module = _SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = _SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = _IdentitySize()

    def forward(self, x: torch.Tensor, sz: List[int]) -> Tuple[torch.Tensor, List[int]]:
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x, sz)


class _SVTRv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [128, 256, 384]
        num_heads = [4, 8, 12]
        mixers = [
            ["Conv"] * 6,
            ["Conv", "Conv", "FGlobal", "Global", "Global", "Global"],
            ["Global"] * 6,
        ]
        kernel_sizes = [[5] * 6, [5] * 6, [-1] * 6]
        num_convs = [[1] * 6, [1] * 6, [-1] * 6]
        sub_k = [[2, 1], [2, 1], [-1, -1]]
        self.out_channels = dims[-1]
        self.pope = _POPatchEmbed(in_channels=3, embed_dim=dims[0], bias=True)
        self.stages = nn.ModuleList(
            [
                _SVTRStage(
                    dim=dims[i],
                    out_dim=dims[i + 1] if i < 2 else 0,
                    mixer=mixers[i],
                    kernel_sizes=kernel_sizes[i],
                    sub_k=sub_k[i],
                    num_heads=num_heads[i],
                    num_conv=num_convs[i],
                    downsample=(i < 2),
                )
                for i in range(3)
            ]
        )
        self.stages.append(_Feat2D())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x


class _EncoderWithSVTR(nn.Module):
    def __init__(self, in_channels: int = 384, dims: int = 256, hidden_dims: int = 256, depth: int = 2):
        super().__init__()
        self.conv1 = _ConvBNLayer(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1), act=_Swish, bias=False)
        self.conv2 = _ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act=_Swish, bias=False)
        self.svtr_block = nn.ModuleList(
            [
                _Block(
                    dim=hidden_dims,
                    num_heads=8,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    act_layer=_Swish,
                    eps=1e-5,
                    prenorm=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = _ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=_Swish, bias=False)
        self.conv4 = _ConvBNLayer(2 * in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1), act=_Swish, bias=False)
        self.conv1x1 = _ConvBNLayer(in_channels // 8, dims, kernel_size=1, act=_Swish, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x.mean(dim=2, keepdim=True), kernel_size=(1, 2))
        h = x
        z = self.conv2(self.conv1(x))
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2).contiguous()
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z).reshape(B, H, W, C).permute(0, 3, 1, 2)
        z = self.conv3(z)
        return self.conv1x1(self.conv4(torch.cat((h, z), dim=1)))


class _CTCDecoder(nn.Module):
    def __init__(self, in_channels: int = 384, out_channels: int = 6625):
        super().__init__()
        self.svtr_encoder = _EncoderWithSVTR(in_channels=in_channels, dims=256, hidden_dims=256, depth=2)
        self.fc = nn.Linear(256, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.svtr_encoder(x).flatten(2).transpose(1, 2)
        return F.softmax(self.fc(x), dim=2)


class _SVTRv2Server(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.encoder = _SVTRv2Encoder()
        self.decoder = _CTCDecoder(in_channels=self.encoder.out_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def load_ppocr_keys(dict_path: Path = _DICT_PATH) -> List[str]:
    chars = ["blank"]
    with open(dict_path, "rb") as fin:
        for line in fin:
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    chars.append(" ")
    return chars


class SVTRv2OCRRecognizer:
    """Lazy-loading pure-PyTorch OpenOCR SVTRv2 Server recognizer."""

    def __init__(self, model_key: str = "svtrv2", use_gpu: bool = True):
        self.model_key = model_key
        self.repo_id = SVTRV2_REPO_ID
        self.use_gpu = bool(use_gpu)
        self._device = "cpu"
        self._chars: List[str] = []
        self._model: Optional[_SVTRv2Server] = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download

        self._device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        self._chars = load_ppocr_keys()

        try:
            weights_path = hf_hub_download(self.repo_id, SVTRV2_WEIGHTS_FILENAME, local_files_only=True)
        except Exception:
            logger.info("Downloading OpenOCR SVTRv2 weights from '%s'...", self.repo_id)
            weights_path = hf_hub_download(self.repo_id, SVTRV2_WEIGHTS_FILENAME)

        model = _SVTRv2Server(out_channels=len(self._chars))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        filtered_sd = {k: v for k, v in raw_sd.items() if "before_gtc" not in k and "gtc_head" not in k}
        model.load_state_dict(filtered_sd, strict=True)
        self._model = model.to(self._device).eval()
        logger.info("OpenOCR SVTRv2 ready on %s.", self._device)

    @staticmethod
    def _preprocess_strip(strip_bgr: np.ndarray) -> torch.Tensor:
        h, w = strip_bgr.shape[:2]
        ratio = w / float(h)
        # Ensure padded width is a multiple of 4 so POPatchEmbed(stride=4) shape matches SubSample1D reshape.
        img_w = ((max(320, int(math.ceil(48 * ratio))) + 3) // 4) * 4
        resized_w = min(img_w, max(4, int(math.ceil(48 * ratio))))
        resized = cv2.resize(strip_bgr, (resized_w, 48), interpolation=cv2.INTER_CUBIC)
        norm = (resized.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        padded = np.zeros((1, 3, 48, img_w), dtype=np.float32)
        padded[0, :, :, :resized_w] = norm
        return torch.from_numpy(padded)

    def _infer_strip(self, strip_bgr: np.ndarray) -> Tuple[str, float]:
        inp = self._preprocess_strip(strip_bgr).to(self._device)
        with torch.no_grad():
            probs = self._model(inp).cpu().numpy()[0]
        pred_idx = probs.argmax(axis=1)
        sel = np.ones(len(pred_idx), dtype=bool)
        sel[1:] = pred_idx[1:] != pred_idx[:-1]
        sel &= pred_idx != 0
        if not sel.any():
            return "", 0.0
        text = "".join(self._chars[i] for i in pred_idx[sel]).strip()
        conf = float(probs.max(axis=1)[sel].mean())
        return text, conf

    def recognize_lines(self, line_crops_bgr: Sequence[np.ndarray], batch_size: int = 1) -> List[Tuple[str, float]]:
        valid: List[Tuple[int, np.ndarray]] = []
        for idx, crop in enumerate(line_crops_bgr):
            if crop is not None and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                valid.append((idx, crop))

        results: List[Tuple[str, float]] = [("", 0.0) for _ in line_crops_bgr]
        if not valid:
            return results

        self._ensure_loaded()
        for pos, orig_crop in valid:
            prepped, stretched = expand_condensed_strip(orig_crop)
            text, conf = self._infer_strip(prepped)
            if not stretched:
                text, conf = refine_condensed_line(orig_crop, text, conf, self._infer_strip)
            results[pos] = (text, conf)
        return results

    def recognize_box(self, image_bgr: np.ndarray, box: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        return recognize_box_lines(image_bgr, box, self.recognize_lines)

    def unload(self):
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            del self._model
        self._model = None
        _release_torch_memory()
