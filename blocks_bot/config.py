import os

from dataclasses import dataclass, field
from typing import Tuple, List, Dict


ADB_PATH = os.environ.get("BLOCKS_ADB_PATH", "adb")
ADB_DEVICE_ID = os.environ.get("BLOCKS_ADB_DEVICE_ID", "")
SCRCPY_PATH = os.environ.get("BLOCKS_SCRCPY_PATH", "scrcpy")
SCRCPY_SERVER_PATH = os.environ.get("BLOCKS_SCRCPY_SERVER_PATH", "scrcpy-server")

GRID_ROWS = 8
GRID_COLS = 8

GRID_TOP_LEFT: Tuple[int, int] = (60, 562)
GRID_BOTTOM_RIGHT: Tuple[int, int] = (1024, 1524)

CELL_W: float = (GRID_BOTTOM_RIGHT[0] - GRID_TOP_LEFT[0]) / GRID_COLS   # ~120.5
CELL_H: float = (GRID_BOTTOM_RIGHT[1] - GRID_TOP_LEFT[1]) / GRID_ROWS   # ~120.25

TRAY_SLOT_CENTERS: List[Tuple[int, int]] = [
    (220, 1764),
    (516, 1764),
    (813, 1764),
]

TRAY_CELL_PITCH: float = 58.0

PIECE_SLOT_HALF: int = 130

DRAG_DURATION_MS: int = 2000

DRAG_LIFT_Y: int = 0

DRAG_ROW_OFFSET: float = 0.2

DRAG_PRE_HOLD_MS: int = 500

DEBUG: bool = True
AUTO_PLAY: bool = False

HEURISTIC_WEIGHTS: Dict[str, float] = {
    "empty_cells":   2.0,
    "holes_penalty": -18.0,
    "bumpiness":     -1.2,
    "near1":         12.0,
    "near2":         4.0,
    "streak_bonus":  20.0,
}

SOLVER_TIME_LIMIT_S: float = 8.0
