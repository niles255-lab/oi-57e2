import json
import os

import numpy as np
from typing import List, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Catalogo de formas validas do jogo (coordenadas normalizadas, origem 0,0)
# ---------------------------------------------------------------------------

def _s(*cells): return frozenset(cells)

PIECE_CATALOG: List[FrozenSet[Tuple[int, int]]] = [
    # 1 bloco
    _s((0,0)),

    # 2 blocos
    _s((0,0),(0,1)),
    _s((0,0),(1,0)),

    # 3 blocos — barras
    _s((0,0),(0,1),(0,2)),
    _s((0,0),(1,0),(2,0)),
    # 3 blocos — cantos (2x2 com 3 blocos)
    _s((0,0),(1,0),(1,1)),
    _s((0,0),(0,1),(1,0)),
    _s((0,0),(0,1),(1,1)),
    _s((0,1),(1,0),(1,1)),

    # 3 blocos — L pequeno (3 na perna longa + 1 na curta = 4 blocos, mas jogo pode ter variação 3 blocos?)
    # Se o jogo tem L de 3 blocos com perna de 3, seria uma linha de 3 + 1 no final = 4 blocos.
    # Mantemos apenas os cantos 2x2 por enquanto.

    # 4 blocos — barras
    _s((0,0),(0,1),(0,2),(0,3)),
    _s((0,0),(1,0),(2,0),(3,0)),
    # 4 blocos — quadrado
    _s((0,0),(0,1),(1,0),(1,1)),
    # 4 blocos — L e J
    _s((0,0),(0,1),(0,2),(1,0)),
    _s((0,0),(0,1),(0,2),(1,2)),
    _s((0,0),(1,0),(2,0),(2,1)),
    _s((0,1),(1,1),(2,0),(2,1)),
    # 4 blocos — S e Z
    _s((0,0),(1,0),(1,1),(2,1)),
    _s((0,1),(1,0),(1,1),(2,0)),
    # 4 blocos — T
    _s((0,0),(0,1),(0,2),(1,1)),
    _s((0,0),(1,0),(1,1),(2,0)),
    _s((0,1),(1,0),(1,1),(1,2)),  # T invertido
    _s((0,0),(0,1),(1,0),(2,0)),

    # 5 blocos — barras
    _s((0,0),(0,1),(0,2),(0,3),(0,4)),
    _s((0,0),(1,0),(2,0),(3,0),(4,0)),
    # 5 blocos — L grande (todas as 4 rotacoes)
    _s((0,0),(0,1),(0,2),(0,3),(1,0)),   # ####/#...
    _s((0,0),(0,1),(0,2),(0,3),(1,3)),   # ####/...#
    _s((0,0),(1,0),(1,1),(1,2),(1,3)),   # #.../####
    _s((0,3),(1,0),(1,1),(1,2),(1,3)),   # ...#/####
    _s((0,0),(1,0),(2,0),(3,0),(3,1)),   # #./#./#./#
    _s((0,1),(1,1),(2,1),(3,0),(3,1)),   # .#/.#/.#/##
    _s((0,0),(0,1),(1,0),(2,0),(3,0)),   # ##/#./#./#.
    _s((0,0),(0,1),(1,1),(2,1),(3,1)),   # ##/.#/.#/.#
    # 5 blocos — P / F
    _s((0,0),(0,1),(1,0),(1,1),(2,0)),
    _s((0,0),(0,1),(1,0),(1,1),(2,1)),
    # 5 blocos — T grande
    _s((0,0),(0,1),(0,2),(1,1),(2,1)),
    _s((0,1),(1,0),(1,1),(1,2),(2,1)),
    # 5 blocos — S/Z grande
    _s((0,0),(0,1),(1,1),(1,2),(1,3)),
    _s((0,1),(0,2),(1,0),(1,1),(1,2)),  # alternativa
    # 5 blocos — canto + barra
    _s((0,0),(0,1),(0,2),(1,2),(2,2)),
    _s((0,0),(1,0),(2,0),(2,1),(2,2)),
]


def _load_visual_piece_shapes() -> List[FrozenSet[Tuple[int, int]]]:
    """Load previously confirmed visual shapes without diamond variants."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "visual_memory", "pieces", "catalog", "index.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle).get("records", [])
    except (OSError, ValueError, TypeError):
        return []
    out = []
    for record in records:
        try:
            cells = frozenset((int(r), int(c)) for r, c in record["shape"])
        except (KeyError, TypeError, ValueError):
            continue
        if cells and cells not in out:
            out.append(cells)
    return out


for _visual_shape in _load_visual_piece_shapes():
    if _visual_shape not in PIECE_CATALOG:
        PIECE_CATALOG.append(_visual_shape)


@dataclass(eq=False)
class Piece:
    id: int
    cells: List[Tuple[int, int]]
    width: int
    height: int
    raw_mask: Optional[np.ndarray] = None
    bitmask: int = 0
    tray_cx: float = 0.0
    tray_cy: float = 0.0
    # Centroide detectado dos blocos em pixels do frame (para o pickup).
    # None = indisponivel (usar centro do slot como fallback).
    pickup_cx: Optional[float] = None
    pickup_cy: Optional[float] = None
    # Celulas cujo bloco possui diamante (resgate localizado da visao).
    # Informacao separada da ocupacao: o diamante NAO conta como bloco extra.
    # Reservado para estrategias futuras (modo normal / modo satoshis).
    rescued_cells: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.cells = [(int(r), int(c)) for r, c in self.cells]
        if len(set(self.cells)) != len(self.cells):
            raise ValueError("piece cells must be unique")
        if any(r < 0 or c < 0 for r, c in self.cells):
            raise ValueError("piece cells must be non-negative")
        if self.cells:
            expected_h = max(r for r, _ in self.cells) + 1
            expected_w = max(c for _, c in self.cells) + 1
            if self.height != expected_h or self.width != expected_w:
                raise ValueError("piece dimensions do not match cells")
        elif self.width != 0 or self.height != 0:
            raise ValueError("empty piece must have zero dimensions")

        # Compatibility cache only. Board placement never shifts this mask,
        # because a piece-local stride cannot represent a board stride safely.
        self.bitmask = 0
        for r, c in self.cells:
            self.bitmask |= 1 << (r * max(self.width, 1) + c)

    @classmethod
    def from_mask(cls, piece_id: int, mask: np.ndarray) -> "Piece":
        raw = mask.copy() if mask.shape == (5, 5) else None
        rows, cols = np.where(mask == 1)
        if len(rows) == 0:
            return cls(id=piece_id, cells=[], width=0, height=0, raw_mask=raw)
        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())
        cells = [(int(r - min_r), int(c - min_c)) for r, c in zip(rows, cols)]
        return cls(
            id=piece_id,
            cells=cells,
            width=max_c - min_c + 1,
            height=max_r - min_r + 1,
            raw_mask=raw,
        )

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return (self.id == other.id and self.cells == other.cells
                and self.bitmask == other.bitmask)

    @property
    def anchor_offset(self) -> Tuple[float, float]:
        return (self.height - 1) / 2.0, (self.width - 1) / 2.0

    def __repr__(self):
        return f"Piece(id={self.id}, {self.width}x{self.height}, blocks={len(self.cells)})"


@dataclass
class Move:
    piece_index: int
    row: int
    col: int
    score: float = 0.0


class Board:
    def __init__(self, rows: int = 8, cols: int = 8):
        if rows <= 0 or cols <= 0:
            raise ValueError("board dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.bitboard: int = 0
        self.combo_streak: int = 0
        self.total_score: int = 0
        # Compatibility masks use the actual board width, never a fixed 8.
        self.row_full_mask: int = (1 << cols) - 1
        self.col_base_mask: int = sum(1 << (r * cols) for r in range(rows))

    def sync_bitboard(self):
        """Rebuild the compatibility cache from the authoritative grid."""
        self.bitboard = 0
        for r, c in zip(*np.where(self.grid == 1)):
            self.bitboard |= 1 << (int(r) * self.cols + int(c))

    def copy(self) -> "Board":
        b = Board(self.rows, self.cols)
        b.grid[:]       = self.grid
        b.combo_streak  = self.combo_streak
        b.total_score   = self.total_score
        b.sync_bitboard()
        return b

    def __str__(self):
        return "\n".join("".join("#" if c else "." for c in row) for row in self.grid)


def is_legal(board: Board, piece: Piece, row: int, col: int) -> bool:
    if not piece.cells:
        return False
    if row < 0 or row + piece.height > board.rows:
        return False
    if col < 0 or col + piece.width > board.cols:
        return False
    return all(board.grid[row + dr, col + dc] == 0
               for dr, dc in piece.cells)


def apply_move(board: Board, piece: Piece, row: int, col: int) -> Tuple["Board", int, int]:
    if not is_legal(board, piece, row, col):
        raise ValueError(f"illegal move: piece={piece.id} row={row} col={col}")
    nb = board.copy()
    for dr, dc in piece.cells:
        nb.grid[row + dr, col + dc] = 1

    rows_cleared = [r for r in range(nb.rows) if np.all(nb.grid[r, :] == 1)]
    cols_cleared = [c for c in range(nb.cols) if np.all(nb.grid[:, c] == 1)]

    lines = len(rows_cleared) + len(cols_cleared)
    for r in rows_cleared:
        nb.grid[r, :] = 0
    for c in cols_cleared:
        nb.grid[:, c] = 0
    nb.sync_bitboard()

    base = len(piece.cells)
    if lines > 0:
        nb.combo_streak += 1
        clear_pts = (lines * (lines + 1) // 2) * 10 * max(1, nb.combo_streak)
    else:
        nb.combo_streak = 0
        clear_pts = 0
    nb.total_score += base + clear_pts
    return nb, lines, base + clear_pts
