#!/usr/bin/env python3
"""
Teste de fluidez do drag (diagnostico apenas).
- Localiza peca na bandeja (prefere slot 0).
- touch down no centro, hold, tour por waypoints, touch up NA ORIGEM.
- NAO coloca peca no tabuleiro. NAO executa jogadas. NAO roda solver.
- NAO altera vision.py/bot.py.
"""

import sys
import os
import time
import datetime
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import adb_io
from vision import read_pieces, detect_board

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logs", "diag_dragfluid")
os.makedirs(OUTDIR, exist_ok=True)

STEP_PX = 25
STEP_SLEEP = 0.05
FREEZE_MS = 200.0


def snap(msg):
    ts = datetime.datetime.now().strftime("%H%M%S")
    path = os.path.join(OUTDIR, f"{ts}_{msg}.png")
    fr = adb_io.capture_frame()
    if fr is not None:
        cv2.imwrite(path, fr)
        print(f"[snap] {os.path.basename(path)} frame={fr.shape[1]}x{fr.shape[0]}", flush=True)
        return fr
    print(f"[snap] FALHOU captura ({msg})", flush=True)
    return None


def main():
    t_start = time.time()
    print("== teste de fluidez do drag ==", flush=True)
    if not adb_io.check_device():
        print("ERRO: dispositivo ausente", flush=True)
        return

    print("[stream] iniciando (observacao, igual ao bot)...", flush=True)
    adb_io.init_scrcpy_stream()
    time.sleep(2.0)

    fr0 = adb_io.capture_frame()
    board0 = detect_board(fr0)
    pieces0 = read_pieces(fr0)
    occ = [p for p in pieces0 if p is not None and p.cells]
    print(f"board: {board0.rows}x{board0.cols} fill={int(board0.grid.sum())}", flush=True)
    for p in occ:
        print(f"  slot{p.id}: {p.width}x{p.height} n={len(p.cells)}", flush=True)
    pick = next((p for p in occ if p.id == 0), occ[0] if occ else None)
    if pick is None:
        print("ABORTADO: bandeja vazia, sem peca para segurar", flush=True)
        adb_io.stop_scrcpy_stream()
        return
    sx, sy = config.TRAY_SLOT_CENTERS[pick.id]
    print(f"peca escolhida: slot {pick.id} centro=({sx},{sy}) "
          f"(S0 {'ok' if pick.id == 0 else 'VAZIO -> usando slot %d' % pick.id})", flush=True)
    sig0 = (board0.rows, board0.cols, board0.grid.tobytes(),
            tuple(sorted((p.id, p.width, p.height, tuple(sorted(p.cells))) for p in occ)))

    snap("01_antes_pickup")

    print("[monkey] iniciando...", flush=True)
    if not adb_io.monkey_start():
        print("ABORTADO: monkey nao iniciou", flush=True)
        adb_io.stop_scrcpy_stream()
        return
    # o servidor monkey precisa de um instante apos o connect antes do
    # primeiro comando (no bot isso ocorre naturalmente via planejamento)
    time.sleep(3.0)

    events = []  # (t, x, y, lat_ms, ok)
    fails = 0

    def move_to(x, y):
        nonlocal fails
        t0 = time.time()
        ok = adb_io.touch_move(int(x), int(y))
        lat = (time.time() - t0) * 1000.0
        events.append((time.time() - t_start, int(x), int(y), lat, ok))
        if not ok:
            fails += 1
        time.sleep(STEP_SLEEP)
        return ok

    # touch down + hold (confirmacao do pickup); 1 retry apos 2s
    # (primeiro comando pos-connect pode falhar com servidor recem-criado)
    t0 = time.time()
    ok = adb_io.touch_down(sx, sy)
    if not ok:
        print("[down] primeira tentativa falhou; reconectando...", flush=True)
        adb_io.monkey_reconnect()
        time.sleep(2.0)
        t0 = time.time()
        ok = adb_io.touch_down(sx, sy)
    print(f"[down] ({sx},{sy}) ok={ok} lat={(time.time()-t0)*1000:.1f}ms", flush=True)
    if not ok:
        print("ABORTADO: touch down falhou 2x", flush=True)
        adb_io.monkey_stop()
        try:
            import subprocess as _sp
            _sp.run([config.ADB_PATH, "-s", config.ADB_DEVICE_ID,
                     "shell", "pkill", "-f", "monkey"],
                    capture_output=True, timeout=10)
        except Exception:
            pass
        adb_io.stop_scrcpy_stream()
        return
    time.sleep(1.0)  # hold p/ confirmar pickup
    snap("02_apos_down_hold")

    # waypoints: perto -> centro -> topo -> lateral -> volta pelo caminho -> origem
    waypoints = [(sx + 60, sy - 40), (540, 1170), (540, 750),
                 (900, 1300), (540, 750), (540, 1170),
                 (sx + 60, sy - 40), (sx, sy)]
    cx, cy = float(sx), float(sy)
    aborted = False
    for wx, wy in waypoints:
        dx, dy = wx - cx, wy - cy
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        steps = max(1, int(dist / STEP_PX))
        for s in range(1, steps + 1):
            if not move_to(cx + dx * s / steps, cy + dy * s / steps):
                print(f"[ERRO] touch_move falhou em ({wx},{wy}) — abortando tour",
                      flush=True)
                aborted = True
                break
        cx, cy = float(wx), float(wy)
        snap(f"03_tour_{wx}_{wy}")
        if aborted:
            break

    # touch up NA ORIGEM (sem colocar no tabuleiro)
    t0 = time.time()
    ok = adb_io.touch_up(sx, sy)
    print(f"[up] ({sx},{sy}) origem ok={ok} lat={(time.time()-t0)*1000:.1f}ms", flush=True)
    time.sleep(1.0)
    fr1 = snap("04_depois_up")

    adb_io.monkey_stop()

    # comparacao antes/depois
    time.sleep(1.0)
    fr2 = adb_io.capture_frame()
    board1 = detect_board(fr2)
    pieces1 = read_pieces(fr2)
    occ1 = [p for p in pieces1 if p is not None and p.cells]
    sig1 = (board1.rows, board1.cols, board1.grid.tobytes(),
            tuple(sorted((p.id, p.width, p.height, tuple(sorted(p.cells))) for p in occ1)))
    print(f"board antes={int(board0.grid.sum())} depois={int(board1.grid.sum())}", flush=True)
    print(f"pecas antes={[ (p.id, len(p.cells)) for p in occ ]} "
          f"depois={[ (p.id, len(p.cells)) for p in occ1 ]}", flush=True)
    print(f"estado identico: {sig0 == sig1}", flush=True)

    # estatisticas do percurso
    lats = [e[3] for e in events if e[4]]
    gaps = []
    for a, b in zip(events, events[1:]):
        gaps.append((b[0] - a[0]) * 1000.0)
    print(f"pontos enviados: {len(events)} | falhas: {fails}", flush=True)
    if lats:
        print(f"latencia touch_move: min={min(lats):.1f}ms media={sum(lats)/len(lats):.1f}ms "
              f"max={max(lats):.1f}ms", flush=True)
    if gaps:
        ng = sum(1 for g in gaps if g > FREEZE_MS)
        print(f"intervalo entre pontos: media={sum(gaps)/len(gaps):.1f}ms max={max(gaps):.1f}ms "
              f"congelamentos(>{FREEZE_MS:.0f}ms): {ng}", flush=True)
        worst = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)[:5]
        for i in worst:
            a, b = events[i], events[i + 1]
            print(f"  gap {gaps[i]:.0f}ms entre ({a[1]},{a[2]}) -> ({b[1]},{b[2]})", flush=True)
    print(f"duracao total: {time.time() - t_start:.1f}s", flush=True)

    adb_io.monkey_stop()
    try:
        import subprocess as _sp
        _sp.run([config.ADB_PATH, "-s", config.ADB_DEVICE_ID,
                 "shell", "pkill", "-f", "monkey"],
                capture_output=True, timeout=10)
        _sp.run([config.ADB_PATH, "forward", "--remove", "tcp:1080"],
                capture_output=True, timeout=10)
    except Exception:
        pass
    adb_io.stop_scrcpy_stream()
    print("fim (nenhuma jogada executada).", flush=True)


if __name__ == "__main__":
    main()
