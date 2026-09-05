#!/usr/bin/env python3
"""
Diagnóstico de UMA jogada (para, não repete, não faz autoplay).

Fluxo: captura -> board(perfil) -> peças -> solver (imprime decisão) ->
[MOVE] (origem/destino/pitch/offsets) -> executa UMA jogada ->
recaptura -> EXPECTED vs ACTUAL -> para.

Uso: python diag_move_once.py
Não escreve no catálogo. Não altera detector/solver.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adb_io
import board_catalog as bc
from vision import detect_board, read_pieces
from model import apply_move
import bot as _bot
from solver import best_sequence


def main():
    print("=== DIAG uma jogada (sem autoplay, sem catalogo) ===")
    if not adb_io.check_device():
        print("sem dispositivo")
        return
    status, det, saved = None, None, None
    fr = adb_io.capture_frame()
    if fr is None:
        print("captura falhou")
        return
    status, det, saved = bc.ensure_board(fr)
    print(f"[BOARD] status={status} perfil={saved['shape'] if saved else None}")
    if status != "known" or saved is None:
        print("sem perfil validado: NADA MOVIDO. pare.")
        return
    profile = saved
    _bot._BOARD_PROFILE = profile

    board = detect_board(fr, profile=profile)
    pieces = read_pieces(fr)
    valid = [p for p in pieces if p is not None and p.cells]
    print(f"[BOARD] {board.rows}x{board.cols} fill={int(board.grid.sum())}")
    for p in valid:
        print(f"  slot {p.id}: {p.width}x{p.height} {sorted(p.cells)}")
    if not valid:
        print("bandeja vazia: nada a fazer.")
        return
    seq = best_sequence(board, pieces)
    if not seq:
        print("solver sem jogadas.")
        return
    m = seq[0]
    piece = next(p for p in valid if p.id == m.piece_index)
    print(f"[SOLVER] slot {m.piece_index} -> celula ({m.row},{m.col}) "
          f"({len(seq)} na sequencia)")

    sx, sy = __import__("config").TRAY_SLOT_CENTERS[piece.id]
    ex, ey = _bot._target_ey(m, piece, board)
    cw, ch = _bot._cell_size(board)
    import config
    print("[MOVE]")
    print(f"piece={piece.id}")
    print(f"shape={sorted(piece.cells)}")
    print(f"source=({sx},{sy})")
    print(f"target_cell=({m.row},{m.col})")
    print(f"target_pixel=({ex},{ey})")
    print(f"board={board.rows}x{board.cols}")
    print(f"pitch=({cw:.2f},{ch:.2f})")
    print(f"offset=(lift={config.DRAG_LIFT_Y},rowoff={config.DRAG_ROW_OFFSET})")

    exp_board, exp_lines, _ = apply_move(board, piece, m.row, m.col)
    exp_cells = set(zip(*__import__("numpy").where(exp_board.grid == 1)))

    print("[EXEC] uma jogada via bot.execute_move ...")
    ok = _bot.execute_move(m, piece, before_frame=fr, board=board)
    print(f"adb_return={ok}")

    import time
    time.sleep(1.0)
    adb_io.wait_stable(timeout=3.0, poll=0.15, required_stable=2)
    fr2 = adb_io.capture_frame()
    if fr2 is None:
        print("sem frame pos-jogada.")
        return
    board2 = detect_board(fr2, profile=profile)
    act_cells = set(zip(*__import__("numpy").where(board2.grid == 1)))
    print(f"EXPECTED ocupadas={len(exp_cells)} (linhas limpas={exp_lines})")
    print(f"ACTUAL   ocupadas={len(act_cells)}")
    only_exp = sorted(exp_cells - act_cells)
    only_act = sorted(act_cells - exp_cells)
    print(f"so-esperado={only_exp}")
    print(f"so-real={only_act}")
    _transport = "swipe"
    _src = (sx, sy)
    print(f"TRANSPORT = {_transport}")
    print(f"SOURCE = {_src}")
    print(f"TARGET = ({m.row},{m.col}) -> ({ex},{ey})")
    print(f"EXPECTED = {len(exp_cells)}")
    print(f"ACTUAL = {len(act_cells)}")
    print(f"MISSING = {only_exp}")
    print(f"EXTRA = {only_act}")
    if not only_exp and not only_act:
        print("[MOVE OK] esperado == real")
        print("RESULT = EFFECTIVE")
    else:
        _bef = set(zip(*__import__("numpy").where(board.grid == 1)))
        _act = set(zip(*__import__("numpy").where(board2.grid == 1)))
        print("[MOVE ERROR] esperado != real (sem retry, parado)")
        print("RESULT =", "NOOP" if _act == _bef else "ERROR")
    print("FIM (1 jogada, parado)")


if __name__ == "__main__":
    main()
