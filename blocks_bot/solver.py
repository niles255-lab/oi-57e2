from typing import List, Optional, Tuple
import time

import numpy as np

from . import config
from .model import Board, Piece, Move, apply_move, is_legal


def evaluate(board: Board) -> float:
    """Score a board using the grid as the only source of truth."""
    weights = config.HEURISTIC_WEIGHTS
    grid = board.grid.astype(bool)
    rows, cols = grid.shape
    occupied = int(grid.sum())
    score = (rows * cols - occupied) * weights["empty_cells"]

    empty = ~grid
    holes = 0
    if cols >= 3:
        holes += int(np.logical_and(empty[:, 1:-1],
                                    grid[:, :-2] & grid[:, 2:]).sum())
    if rows >= 3:
        holes += int(np.logical_and(empty[1:-1, :],
                                    grid[:-2, :] & grid[2:, :]).sum())
    score += holes * weights["holes_penalty"]

    heights = []
    for c in range(cols):
        occupied_rows = np.flatnonzero(grid[:, c])
        heights.append(rows - int(occupied_rows[0]) if occupied_rows.size else 0)
    bumpiness = sum(abs(heights[i] - heights[i + 1])
                    for i in range(len(heights) - 1))
    score += bumpiness * weights["bumpiness"]

    near1 = near2 = 0
    for r in range(rows):
        empty_count = cols - int(grid[r, :].sum())
        if empty_count == 1:
            near1 += 1
        elif empty_count == 2:
            near2 += 1
    for c in range(cols):
        empty_count = rows - int(grid[:, c].sum())
        if empty_count == 1:
            near1 += 1
        elif empty_count == 2:
            near2 += 1

    score += near1 * weights["near1"]
    score += near2 * weights["near2"]
    score += board.combo_streak * weights["streak_bonus"]
    return float(score)


_PARTIAL_PENALTY = 350.0


def _dfs(board: Board, pieces: List[Piece], memo, deadline):
    key = (
        board.rows, board.cols, board.grid.tobytes(),
        board.combo_streak, board.total_score,
        tuple(sorted((p.id, tuple(sorted(p.cells))) for p in pieces)),
    )
    if key in memo:
        return memo[key]
    if time.monotonic() >= deadline:
        return float("-inf"), None
    if not pieces:
        result = (board.total_score + evaluate(board), [])
        memo[key] = result
        return result

    best_score = float("-inf")
    best_seq: Optional[List[Move]] = None

    for i, piece in enumerate(pieces):
        remaining = pieces[:i] + pieces[i + 1:]
        for row in range(board.rows - piece.height + 1):
            for col in range(board.cols - piece.width + 1):
                if not is_legal(board, piece, row, col):
                    continue
                next_board, _, _ = apply_move(board, piece, row, col)
                child_score, child_seq = _dfs(next_board, remaining, memo, deadline)
                if child_score == float("-inf"):
                    score = (next_board.total_score + evaluate(next_board)
                             - _PARTIAL_PENALTY * len(remaining))
                    candidate = [Move(piece_index=piece.id, row=row, col=col)]
                else:
                    score = child_score
                    candidate = [Move(piece_index=piece.id, row=row, col=col)] + (child_seq or [])

                if score > best_score:
                    best_score = score
                    best_seq = candidate

    result = (best_score, best_seq)
    if time.monotonic() < deadline:
        memo[key] = result
    return result


def _valid_pieces(pieces: List[Optional[Piece]]) -> List[Piece]:
    return [p for p in pieces if p is not None and p.cells]


def best_move(board: Board, pieces: List[Optional[Piece]]) -> Optional[Move]:
    valid = _valid_pieces(pieces)
    if not valid:
        return None
    _, sequence = _dfs(board.copy(), valid, {},
                       time.monotonic() + config.SOLVER_TIME_LIMIT_S)
    return sequence[0] if sequence else None


def best_sequence(board: Board, pieces: List[Optional[Piece]]) -> List[Move]:
    """Return a legal sequence for the current board and available pieces."""
    valid = _valid_pieces(pieces)
    if not valid:
        return []
    _, sequence = _dfs(board.copy(), valid, {},
                       time.monotonic() + config.SOLVER_TIME_LIMIT_S)
    return sequence or []
