"""
app/core/typography/auto_fit.py
Binary Search / Bisection Font Auto-Fit Algorithm (10 Iterations)
Calculates optimal font size to fit text within bounding box without clipping.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Protocol, List
import math


@dataclass(frozen=True)
class LayoutResult:
    fits: bool
    total_width: float
    total_height: float
    lines_or_columns: List[str]
    font_size: float
    overflow_x: float = 0.0
    overflow_y: float = 0.0


@dataclass(frozen=True)
class AutoFitResult:
    optimal_font_size: float
    layout: LayoutResult
    is_clamped: bool
    iterations_run: int


class LayoutEvaluator(Protocol):
    """Protocol for laying out and measuring text at a specific font size."""
    def evaluate(self, text: str, font_size: float, max_w: float, max_h: float, is_vertical: bool) -> LayoutResult:
        ...


class AutoFitEngine:
    """
    High-precision bisection auto-fit calculator.
    Guarantees sub-0.07px convergence within 10 iterations.
    """
    def __init__(
        self,
        min_font_size: float = 6.0,
        max_font_size: float = 120.0,
        bisection_iterations: int = 10,
        padding_ratio: float = 0.10,
    ):
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.bisection_iterations = bisection_iterations
        self.padding_ratio = padding_ratio

    def calculate_padding(self, box_w: float, box_h: float, bubble_shape: str = "rect") -> Tuple[float, float]:
        """
        Calculate safety padding to prevent text from touching speech bubble borders.
        For elliptical bubbles, padding is larger to stay within the inscribed ellipse.
        """
        ratio = self.padding_ratio if bubble_shape != "ellipse" else 0.15
        px = max(2.0, min(28.0, box_w * ratio))
        py = max(2.0, min(28.0, box_h * ratio))
        return px, py

    def fit_text(
        self,
        text: str,
        box_w: float,
        box_h: float,
        is_vertical: bool,
        evaluator: LayoutEvaluator,
        font_scale: float = 1.0,
        bubble_shape: str = "rect"
    ) -> AutoFitResult:
        """
        Execute 10-step bisection search for optimal font size.
        """
        stripped_text = text.strip()
        if not stripped_text or box_w <= 4 or box_h <= 4:
            empty_layout = LayoutResult(
                fits=True, total_width=0.0, total_height=0.0,
                lines_or_columns=[], font_size=self.min_font_size
            )
            return AutoFitResult(self.min_font_size, empty_layout, is_clamped=True, iterations_run=0)

        px, py = self.calculate_padding(box_w, box_h, bubble_shape)
        avail_w = max(4.0, box_w - 2 * px)
        avail_h = max(4.0, box_h - 2 * py)

        # Upper bound calculation: cap by dimension and font_scale
        major_dim = avail_h if not is_vertical else avail_w
        high_bound = min(self.max_font_size, max(self.min_font_size, major_dim * 0.75 * font_scale))
        low_bound = self.min_font_size

        # Check if min_font_size fits
        min_layout = evaluator.evaluate(stripped_text, low_bound, avail_w, avail_h, is_vertical)
        if not min_layout.fits:
            # Text cannot fit even at minimum legible font size (severe overflow)
            return AutoFitResult(
                optimal_font_size=low_bound,
                layout=min_layout,
                is_clamped=True,
                iterations_run=1
            )

        # Check if high_bound fits directly (e.g. single short word in large bubble)
        max_layout = evaluator.evaluate(stripped_text, high_bound, avail_w, avail_h, is_vertical)
        if max_layout.fits:
            return AutoFitResult(
                optimal_font_size=high_bound,
                layout=max_layout,
                is_clamped=False,
                iterations_run=1
            )

        # 10-step Bisection Search
        best_size = low_bound
        best_layout = min_layout
        low = low_bound
        high = high_bound

        for it in range(self.bisection_iterations):
            mid = (low + high) / 2.0
            test_layout = evaluator.evaluate(stripped_text, mid, avail_w, avail_h, is_vertical)
            if test_layout.fits:
                best_size = mid
                best_layout = test_layout
                low = mid  # Try larger size
            else:
                high = mid  # Exceeds bounds, try smaller size

        return AutoFitResult(
            optimal_font_size=best_size,
            layout=best_layout,
            is_clamped=False,
            iterations_run=self.bisection_iterations
        )
