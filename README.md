# 🌌 AetherLens - Automatic Manga Translation System / 自动漫画翻译系统

<p align="center">
  <a href="#-aetherlens---自动漫画翻译系统">简体中文</a> | 
  <a href="#-aetherlens---automatic-manga-translation-system">English</a>
</p>

---

# 🇨🇳 AetherLens - 自动漫画翻译系统

AetherLens 是一款专为漫画、条漫（Webtoon）打造的**高精度本地化翻译与图像修复系统**。通过结合本地部署的 **PaddleOCR 3.x** 图像文字识别引擎与主流的 AI 大语言模型（LLM），系统能自动提取漫画中的文本（完美支持竖排与日语）、擦除原文并对背景进行高保真羽化修复，最终将翻译后的文本完美渲染回原图位置。

## 🚀 核心特性

- **高精度本地 OCR**：集成 **PaddleOCR 3.x** 引擎，针对日语漫画、竖排文本及复杂排版进行了深度对齐，识别率显著领先于传统 OCR。
- **智能硬件自适应**：自动检测本地 GPU 设备。针对 GTX 1080 Ti 等 Pascal 架构老显卡在 CUDA 12 下的底层计算乱码 Bug，系统会自动、安全地切换为 CPU 推理，确保 100% 识别率；其余显卡自动启用 GPU 硬件加速。
- **高保真图像修复 (Inpainting)**：
  - 支持 **OpenCV 智能羽化算法** 极速修补。
  - 支持集成 **LaMa 深度学习修复模型**（`simple-lama-inpainting`），实现背景纹理无缝补全。
  - 动态计算对话框和手写拟声词（Onomatopoeia）的膨胀掩码，结合双边滤波，保留漫画原始背景线条。
- **多模型翻译对接**：支持一键接入 **Gemini**、**OpenAI**、**DeepSeek** 等 API，支持自定义代理端点。
- **极简一键部署**：提供 Windows 下的一键式环境搭建与启动脚本，小白也能轻松部署。

---

## 🛠️ 环境要求

- **操作系统**：Windows 10 / 11 (64-bit)
- **Node.js**：v18 或 v20 LTS (推荐)
- **Python**：v3.12 (推荐)
- **GPU 驱动**（如需显卡加速）：建议升级至最新 NVIDIA 官方驱动（支持 CUDA 12+）

---

## ⚡ 一键部署与启动 (推荐)

项目根目录下已为您内置了自动化部署脚本：

1. 双击运行 **[one_click_deploy.bat](file:///D:/baidu/download/web/one_click_deploy.bat)**。
2. 脚本将自动完成以下操作：
   - 检查并配置 Node.js 与 Python 环境。
   - 自动安装前端所需的 Node 依赖（React, Vite 等）。
   - 自动安装后端所需的 Python 依赖（Flask, OpenCV, NumPy, PaddleOCR 等）。
   - 自动在独立窗口中拉起 **OCR 后端服务** 与 **Vite 前端 Web 应用**。
3. 部署成功后：
   - 浏览器将自动打开前端页面：[http://localhost:5173](http://localhost:5173)
   - 本地 OCR 后端服务运行在：`http://127.0.0.1:5000`

*项目还提供了自动清理与卸载脚本 **[uninstall.bat](file:///D:/baidu/download/web/uninstall.bat)**，可按需一键清理项目依赖与本地模型缓存文件。*

---

## 🔧 手动分步启动

如果您希望手动控制服务的启动，可以分别执行以下命令：

### 1. 启动本地 OCR 后端服务
首先确保已安装依赖包：
```bash
pip install flask flask-cors opencv-python numpy paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
然后运行：
```bash
python ocr_server.py
```
或直接双击 **[start_ocr_server.bat](file:///D:/baidu/download/web/start_ocr_server.bat)**。

### 2. 启动前端开发服务器
```bash
npm install
npm run dev
```

---

## ⚙️ 常见问题与排查 (FAQ)

### Q: 为什么我用 1080 Ti 等显卡运行时，文字识别出来的全是不知所云的乱码？
- **原因**：这是 PaddlePaddle 3.x 官方在 CUDA 12 环境下针对 NVIDIA Pascal 架构（GTX 10 系列，如 1080Ti/1070/1060 等）的底层算子 Bug。GPU 计算矩阵卷积时会产生计算漂移，导致识别出乱码且置信度为零。
- **解决方法**：系统目前已具备**智能显卡黑名单检测**。如果检测到您的显卡属于 10 系列或更早架构，系统会**自动将推理设备降级为 CPU 运行**，不仅完全避免了乱码，而且推理速度依然极快（约 1 秒/页）。如果您更换为 RTX 20/30/40 等新显卡，系统将自动恢复 GPU 硬件加速。

### Q: 如何启用 LaMa 深度学习图像修复？
- 系统默认使用 OpenCV 修复，已具备出色的羽化融合效果。若想获得更完美的背景填充，可在 Python 环境中安装：
  ```bash
  pip install simple-lama-inpainting
  ```
  安装后重启后端，系统检测到该库后会自动加载 LaMa 修复模型。

---

<br>
<br>

---

# 🇺🇸 AetherLens - Automatic Manga Translation System

AetherLens is a **high-precision localization, translation, and image restoration system** custom-built for manga and webtoons. By combining a locally deployed **PaddleOCR 3.x** engine with cutting-edge Large Language Models (LLMs), the system automates text extraction (expertly handling vertical Japanese text), background-aware text erasing, high-fidelity inpainting, and precise target text re-rendering.

## 🚀 Key Features

- **High-Precision Local OCR**: Integrated with **PaddleOCR 3.x**, deeply optimized for vertical text, hand-drawn fonts, and dense manga layouts.
- **Smart Hardware Adaptive Engine**: Dynamically queries local GPU models. For older Pascal GPUs (such as GTX 1080 Ti) which suffer from calculation bugs under CUDA 12, the system automatically falls back to CPU to guarantee 100% correct recognition. Modern GPUs (RTX 20/30/40 series) automatically run with full GPU acceleration.
- **High-Fidelity Inpainting**:
  - Superfast OpenCV-based edge-feathered flat-fills.
  - Deep-learning-based **LaMa Inpainting** (`simple-lama-inpainting`) for seamless texture restoration.
  - Dynamic mask dilation (differentiating between dialogues and background sound effects) and Bilateral Filtering to preserve original artwork details.
- **Multi-LLM Integration**: Seamlessly connect to **Gemini**, **OpenAI**, or **DeepSeek** APIs, with support for custom API endpoints.
- **One-Click Deployer**: Pre-configured batch files to set up the environment and launch services instantly.

---

## 🛠️ Requirements

- **OS**: Windows 10 / 11 (64-bit)
- **Node.js**: v18 or v20 LTS (Recommended)
- **Python**: v3.12 (Recommended)
- **NVIDIA GPU Driver** (For GPU acceleration): Latest official driver supporting CUDA 12+

---

## ⚡ One-Click Deployment (Recommended)

An automated deployer script is pre-built in the project root:

1. Double-click to run **[one_click_deploy.bat](file:///D:/baidu/download/web/one_click_deploy.bat)**.
2. The script will automatically:
   - Check and configure your Node.js and Python environments.
   - Install all frontend dependencies (React, Vite, etc.).
   - Install backend Python dependencies (Flask, OpenCV, NumPy, PaddleOCR, etc.).
   - Launch the **OCR Backend Server** and **Vite Frontend Dev Server** in separate windows.
3. Once completed:
   - The Web UI will open at: [http://localhost:5173](http://localhost:5173)
   - The local OCR API is hosted at: `http://127.0.0.1:5000`

*An uninstaller script **[uninstall.bat](file:///D:/baidu/download/web/uninstall.bat)** is also provided to clean up dependencies and downloaded model cache files.*

---

## 🔧 Step-by-Step Manual Launch

If you prefer starting the services manually:

### 1. Start the OCR Backend Server
Ensure dependencies are installed:
```bash
pip install flask flask-cors opencv-python numpy paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
Run the server:
```bash
python ocr_server.py
```
Or double-click **[start_ocr_server.bat](file:///D:/baidu/download/web/start_ocr_server.bat)**.

### 2. Start the Frontend Dev Server
```bash
npm install
npm run dev
```

---

## ⚙️ Troubleshooting (FAQ)

### Q: Why is the OCR output completely garbled (nonsense characters) on my GTX 1080 Ti?
- **Reason**: This is an upstream bug in PaddlePaddle 3.x's CUDA 12 kernel compiles for NVIDIA Pascal GPUs (GTX 10 series, including 1080Ti/1070/1060, and older). Matrix convolutions on these cards calculate incorrectly, leading to garbage outputs with zero confidence.
- **Solution**: The system has **built-in hardware blacklisting**. If a Pascal/Maxwell GPU is detected, it **automatically falls back to CPU mode**, preventing gibberish output while remaining very fast (~1 second per page). If you upgrade to an RTX card later, the system will automatically enable full GPU acceleration.

### Q: How do I enable the advanced LaMa deep learning inpainter?
- The system defaults to OpenCV flat-fill + feathering, which works great. For neural-network-based texture filling, install `simple-lama-inpainting` in your Python environment:
  ```bash
  pip install simple-lama-inpainting
  ```
  Restart the backend, and it will load LaMa automatically.
