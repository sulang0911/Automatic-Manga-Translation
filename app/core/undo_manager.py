"""
app/core/undo_manager.py
History and Undo/Redo Manager for manga translation workbench.
Provides memory-efficient page state snapshots, undo/redo stacks, and action descriptions.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import copy
import numpy as np


def are_blocks_equal(blocks1: List[Dict[str, Any]], blocks2: List[Dict[str, Any]]) -> bool:
    """Compares two block lists for semantic equality (coords, text, style, type)."""
    if len(blocks1) != len(blocks2):
        return False
    for b1, b2 in zip(blocks1, blocks2):
        # Check core geometry
        for k in ("id", "xmin", "ymin", "xmax", "ymax", "original_text", "translated_text", "type"):
            if b1.get(k) != b2.get(k):
                return False
        # Check style overrides
        for k in ("font_family_override", "font_size_override", "font_bold_override",
                  "stroke_mode_override", "stroke_width_override", "text_color_override",
                  "angle_override"):
            if b1.get(k) != b2.get(k):
                return False
    return True


@dataclass
class PageSnapshot:
    """Snapshot representing the state of a manga page at a point in time."""
    page_path: str
    blocks: List[Dict[str, Any]]
    erased_img: Optional[np.ndarray] = None
    style: Optional[Any] = None
    description: str = "操作"

    @classmethod
    def create(
        cls,
        page_path: str,
        blocks: List[Any],
        erased_img: Optional[np.ndarray] = None,
        style: Optional[Any] = None,
        description: str = "操作"
    ) -> 'PageSnapshot':
        serialized_blocks: List[Dict[str, Any]] = []
        for b in (blocks or []):
            if isinstance(b, dict):
                serialized_blocks.append(copy.deepcopy(b))
            elif hasattr(b, "to_dict"):
                serialized_blocks.append(copy.deepcopy(b.to_dict()))
            else:
                serialized_blocks.append(copy.deepcopy(vars(b)))

        copied_style = copy.deepcopy(style) if style is not None else None

        return cls(
            page_path=page_path or "",
            blocks=serialized_blocks,
            erased_img=erased_img.copy() if (erased_img is not None and hasattr(erased_img, "copy")) else None,
            style=copied_style,
            description=description
        )


class UndoManager:
    """
    Manages undo and redo stacks for user operations.
    Supports bounded stack depth to conserve memory.
    """
    def __init__(self, max_depth: int = 50):
        self.max_depth = max(1, int(max_depth))
        self._undo_stack: List[PageSnapshot] = []
        self._redo_stack: List[PageSnapshot] = []

    def push(self, snapshot: PageSnapshot) -> None:
        """Pushes a pre-action state onto the undo stack and invalidates redo stack."""
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def get_undo_description(self) -> str:
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    def get_redo_description(self) -> str:
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""

    def undo(self, current_snapshot: PageSnapshot) -> Optional[PageSnapshot]:
        """
        Pops the previous state from the undo stack, pushes current state to redo stack,
        and returns the state to restore.
        """
        if not self._undo_stack:
            return None
        target = self._undo_stack.pop()
        # Associate current state with the action being reversed so redo knows what it will do
        redo_snapshot = PageSnapshot.create(
            page_path=current_snapshot.page_path,
            blocks=current_snapshot.blocks,
            erased_img=current_snapshot.erased_img,
            style=current_snapshot.style,
            description=target.description
        )
        self._redo_stack.append(redo_snapshot)
        if len(self._redo_stack) > self.max_depth:
            self._redo_stack.pop(0)
        return target

    def redo(self, current_snapshot: PageSnapshot) -> Optional[PageSnapshot]:
        """
        Pops the state from the redo stack, pushes current state to undo stack,
        and returns the state to restore.
        """
        if not self._redo_stack:
            return None
        target = self._redo_stack.pop()
        undo_snapshot = PageSnapshot.create(
            page_path=current_snapshot.page_path,
            blocks=current_snapshot.blocks,
            erased_img=current_snapshot.erased_img,
            style=current_snapshot.style,
            description=target.description
        )
        self._undo_stack.append(undo_snapshot)
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
        return target

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
