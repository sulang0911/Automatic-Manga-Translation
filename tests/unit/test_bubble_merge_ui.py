"""
tests/unit/test_bubble_merge_ui.py
Unit tests verifying the interactive bubble merge and context menu features:
- InspectorPanel merge_prev_btn and merge_next_btn
- _merge_blocks joins geometry, original_text, and translated_text correctly
- CanvasView and BubbleItem signals for merge and delete
"""
import pytest
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.canvas.items.bubble_item import BubbleItem
from app.ui.canvas.view import MangaCanvasView


def test_inspector_panel_merge_blocks_logic(qapp):
    panel = InspectorPanel()
    assert hasattr(panel, "merge_prev_btn")
    assert hasattr(panel, "merge_next_btn")

    # Setup two split blocks (like user sample: top half and bottom half)
    b1 = {
        "id": "block_top",
        "xmin": 10.0, "ymin": 10.0, "xmax": 90.0, "ymax": 30.0,
        "original_text": "Emergency Sexual Service Submissive Seat",
        "translated_text": "紧急性服务顺从座位",
        "type": "bubble"
    }
    b2 = {
        "id": "block_bottom",
        "xmin": 12.0, "ymin": 31.0, "xmax": 88.0, "ymax": 65.0,
        "original_text": "is a special seat implemented recently.",
        "translated_text": "是最近设立的特殊座位。",
        "type": "bubble"
    }

    panel.set_blocks([b1, b2])
    assert len(panel.current_blocks) == 2

    # Track signals
    reordered_payloads = []
    rerender_called = []
    panel.sig_blocks_reordered.connect(lambda blist: reordered_payloads.append(list(blist)))
    panel.sig_re_render_requested.connect(lambda: rerender_called.append(True))

    # Select first block and merge with next
    panel.select_block_by_id("block_top")
    panel._on_merge_next_clicked()

    # Verify merged result
    assert len(panel.current_blocks) == 1
    merged = panel.current_blocks[0]
    assert merged["id"] == "block_top"
    assert merged["xmin"] == 10.0
    assert merged["ymin"] == 10.0
    assert merged["xmax"] == 90.0
    assert merged["ymax"] == 65.0
    assert "Emergency Sexual Service Submissive Seat\nis a special seat implemented recently." in merged["original_text"]
    assert "紧急性服务顺从座位" in merged["translated_text"]
    assert "特殊座位" in merged["translated_text"]

    # Verify signals emitted
    assert len(reordered_payloads) == 1
    assert len(rerender_called) == 1


def test_canvas_bubble_item_merge_signals(qapp):
    b = {"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 50.0}
    item = BubbleItem(b, 1000, 1000)

    assert hasattr(item.signals, "merge_prev_requested")
    assert hasattr(item.signals, "merge_next_requested")
    assert hasattr(item.signals, "delete_requested")

    view = MangaCanvasView()
    assert hasattr(view, "sig_bubble_merge_prev")
    assert hasattr(view, "sig_bubble_merge_next")
    assert hasattr(view, "sig_bubble_delete")
