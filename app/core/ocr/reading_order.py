"""
app/core/ocr/reading_order.py
2D spatial sorting engine supporting:
1) Manga RTL (Right-to-Left, Top-to-Bottom panel tiers)
2) Webtoon TTB (Top-to-Bottom continuous vertical flow)
3) Western Comic LTR (Left-to-Right, Top-to-Bottom)
"""
from typing import List
from app.core.models import TranslationBlock, ReadingOrderMode


def sort_reading_order(
    blocks: List[TranslationBlock],
    mode: str = ReadingOrderMode.MANGA_RTL.value,
    row_overlap_ratio: float = 0.35,
    row_height_fraction: float = 0.08
) -> List[TranslationBlock]:
    """
    Sorts translation blocks into chronological narrative reading order
    and assigns 1-indexed reading_order properties.
    """
    if not blocks or len(blocks) <= 1:
        for idx, b in enumerate(blocks, start=1):
            b.reading_order = idx
        return blocks

    if mode == ReadingOrderMode.WEBTOON_TTB.value:
        return _sort_webtoon_ttb(blocks)
    elif mode == ReadingOrderMode.WESTERN_LTR.value:
        return _sort_manga_or_western(
            blocks, is_rtl=False, overlap_ratio=row_overlap_ratio, height_fraction=row_height_fraction
        )
    else:
        # Default: Manga RTL
        return _sort_manga_or_western(
            blocks, is_rtl=True, overlap_ratio=row_overlap_ratio, height_fraction=row_height_fraction
        )


def _sort_webtoon_ttb(blocks: List[TranslationBlock]) -> List[TranslationBlock]:
    """
    Webtoons are continuous vertical strips: primary sort strictly by ascending Y,
    secondary sort by ascending X (Left-to-Right) for side-by-side bubbles.
    """
    sorted_blocks = sorted(blocks, key=lambda b: (b.ymin, b.xmin))
    for idx, b in enumerate(sorted_blocks, start=1):
        b.reading_order = idx
    return sorted_blocks


def _sort_manga_or_western(
    blocks: List[TranslationBlock],
    is_rtl: bool = True,
    overlap_ratio: float = 0.35,
    height_fraction: float = 0.08
) -> List[TranslationBlock]:
    """
    Groups bounding boxes into horizontal panel tiers via interval clustering,
    then sorts columns Right-to-Left (manga) or Left-to-Right (western).
    """
    # 1. Initial ordering by vertical center
    sorted_by_y = sorted(blocks, key=lambda b: (b.ymin + b.ymax) / 2.0)

    # 2. Cluster into panel tiers
    tiers: List[List[TranslationBlock]] = []
    for block in sorted_by_y:
        assigned = False
        b_mid = (block.ymin + block.ymax) / 2.0
        b_h = max(0.5, block.ymax - block.ymin)

        for tier in tiers:
            # Calculate tier vertical bounds
            tier_ymin = min(item.ymin for item in tier)
            tier_ymax = max(item.ymax for item in tier)
            tier_mid = (tier_ymin + tier_ymax) / 2.0
            tier_h = max(0.5, tier_ymax - tier_ymin)

            # Compute interval overlap
            overlap = max(0.0, min(block.ymax, tier_ymax) - max(block.ymin, tier_ymin))
            min_h = min(b_h, tier_h)
            rel_overlap = overlap / min_h

            # Belonging condition: vertical overlap > ratio or vertical center distance within threshold
            if rel_overlap > overlap_ratio or abs(b_mid - tier_mid) < (height_fraction * 100.0):
                tier.append(block)
                assigned = True
                break

        if not assigned:
            tiers.append([block])

    # 3. Sort tiers Top-to-Bottom, and sort items within each tier
    ordered_result: List[TranslationBlock] = []
    # Sort tiers by tier median Y
    tiers.sort(key=lambda t: sum((b.ymin + b.ymax) / 2.0 for b in t) / len(t))

    for tier in tiers:
        if is_rtl:
            # Japanese Manga: Right to Left (descending X center), secondary top to bottom (ascending Y)
            tier.sort(key=lambda b: (-((b.xmin + b.xmax) / 2.0), b.ymin))
        else:
            # Western Comic: Left to Right (ascending X center), secondary top to bottom (ascending Y)
            tier.sort(key=lambda b: (((b.xmin + b.xmax) / 2.0), b.ymin))
        ordered_result.extend(tier)

    # 4. Assign 1-indexed reading order
    for idx, b in enumerate(ordered_result, start=1):
        b.reading_order = idx

    return ordered_result
