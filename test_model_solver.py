import unittest

import numpy as np

from model import Board, Piece, apply_move, is_legal
from solver import best_sequence


def single(piece_id=0):
    return Piece(piece_id, [(0, 0)], 1, 1)


class BoardModelTests(unittest.TestCase):
    def test_wide_board_cells_have_distinct_cache_bits(self):
        board = Board(11, 11)
        board.grid[0, 8] = 1
        board.grid[1, 0] = 1
        board.sync_bitboard()
        self.assertEqual(board.bitboard.bit_count(), 2)
        self.assertFalse(is_legal(board, single(), 0, 8))
        self.assertFalse(is_legal(board, single(), 1, 0))
        self.assertTrue(is_legal(board, single(), 0, 7))

    def test_apply_move_uses_real_width_for_11x11(self):
        board = Board(11, 11)
        piece = Piece(0, [(0, 0), (0, 1)], 2, 1)
        next_board, lines, _ = apply_move(board, piece, 4, 9)
        self.assertEqual(lines, 0)
        self.assertEqual(set(zip(*np.where(next_board.grid == 1))), {(4, 9), (4, 10)})
        self.assertEqual(next_board.bitboard.bit_count(), 2)

    def test_row_and_column_clear_simultaneously(self):
        board = Board(11, 11)
        board.grid[0, 1:] = 1
        board.grid[1:, 0] = 1
        board.sync_bitboard()
        next_board, lines, _ = apply_move(board, single(), 0, 0)
        self.assertEqual(lines, 2)
        self.assertTrue(np.all(next_board.grid[0, :] == 0))
        self.assertTrue(np.all(next_board.grid[:, 0] == 0))
        next_board.sync_bitboard()
        self.assertEqual(next_board.bitboard.bit_count(), int(next_board.grid.sum()))

    def test_illegal_move_is_rejected_before_mutation(self):
        board = Board(11, 11)
        board.grid[6, 2] = 1
        board.sync_bitboard()
        piece = Piece(2, [(0, 0), (0, 1)], 2, 1)
        with self.assertRaises(ValueError):
            apply_move(board, piece, 6, 2)
        self.assertEqual(int(board.grid.sum()), 1)

    def test_solver_moves_are_legal_on_wide_board(self):
        board = Board(11, 11)
        board.grid[0, 8] = 1
        board.grid[10, 10] = 1
        board.sync_bitboard()
        pieces = [
            Piece(0, [(0, 0), (0, 1)], 2, 1),
            Piece(1, [(0, 0), (1, 0)], 1, 2),
        ]
        sequence = best_sequence(board, pieces)
        self.assertTrue(sequence)
        current = board
        by_id = {p.id: p for p in pieces}
        for move in sequence:
            piece = by_id[move.piece_index]
            self.assertTrue(is_legal(current, piece, move.row, move.col))
            current, _, _ = apply_move(current, piece, move.row, move.col)

    def test_piece_dimensions_are_validated(self):
        with self.assertRaises(ValueError):
            Piece(0, [(0, 0), (1, 0)], 2, 1)


if __name__ == "__main__":
    unittest.main()
