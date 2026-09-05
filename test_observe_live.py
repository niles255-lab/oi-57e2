#!/usr/bin/env python3
"""
Observador ao vivo: roda o bot normalmente + overlay de visao em janela
separada, sem sobrepor a regiao de captura. Nao altera bot/detector/solver.
Uso: python test_observe_live.py [segundos]   (padrao 180)
"""

import sys
import os
import re
import time
import datetime
import subprocess
import ctypes
import cv2
import numpy as np
import mss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from vision import (detect_board, read_pieces, _build_signal_map,
                    _snap_to_catalog, _normalize, _max_run,
                    _rescue_maps, _complete_winner_with_diamonds,
                    NATIVE_DEVICE_W, NATIVE_DEVICE_H)

PROJECT = os.path.dirname(os.path.abspath(__file__))
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 180
ADB = config.ADB_PATH
DEVICE = config.ADB_DEVICE_ID
OUTDIR = os.path.join(PROJECT, "logs", "diag_observe")
os.makedirs(OUTDIR, exist_ok=True)

user32 = ctypes.windll.user32


class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def find_win(title, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.3)
    return 0


def win_rect(hwnd):
    r = Rect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


def client_rect(hwnd):
    r = Rect()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    w, h = r.right - r.left, r.bottom - r.top
    pt = Point(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return (pt.x, pt.y, w, h)


def overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def mirror_winner(sig, W, H, sx, sy, sc, maps, slot_index=0):
    """Espelho fiel da busca atual (52-64, peso 1.0, strict, ±8+zero,
    completude via helper compartilhado)."""
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    best, best_score = None, float("-inf")
    for pitch in np.arange(52.0 * sc, 64.0 * sc, 1.0 * sc):
        half = int(pitch / 2) + 2
        base = np.arange(max(-half, -8.0), min(half, 8.0) + 1, 3, dtype=np.float64)
        dxa = np.union1d(base, np.array([0.0]))
        dya = dxa
        xf = sx + dxa[:, None] + cc[None, :] * pitch
        yf = sy + dya[:, None] + rr[None, :] * pitch
        xi = np.round(xf).astype(np.int32)
        yi = np.round(yf).astype(np.int32)
        xo = (xi < 0) | (xi >= W)
        yo = (yi < 0) | (yi >= H)
        xc = np.clip(xi, 0, W - 1)
        yc = np.clip(yi, 0, H - 1)
        va = sig[yc[np.newaxis, :, :, np.newaxis],
                 xc[:, np.newaxis, np.newaxis, :]].astype(np.float64)
        va[(xo[:, np.newaxis, np.newaxis, :] | yo[np.newaxis, :, :, np.newaxis])] = 0.0
        bits = va >= 0.60
        for idx in np.argwhere(bits.any(axis=(-1, -2))):
            i, j = int(idx[0]), int(idx[1])
            b = bits[i, j]
            v = va[i, j]
            rw = np.where(b.any(axis=1))[0]
            cw_ = np.where(b.any(axis=0))[0]
            if rw.size == 0 or cw_.size == 0:
                continue
            r0, r1 = int(rw[0]), int(rw[-1]) + 1
            c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
            bb, bv = b[r0:r1, c0:c1], v[r0:r1, c0:c1]
            n = int(bb.sum())
            if n < 2 or n > 5:
                continue
            ms = float(bv[bb].mean())
            if ms < 0.55:
                continue
            bp = np.pad(bb.astype(np.int8), 1)
            nb = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            sup = int(((nb > 0) & bb).sum())
            run = _max_run(bb)
            eib = (r1 - r0) * (c1 - c0) - n
            det = [(r, c) for r in range(r1 - r0) for c in range(c1 - c0) if bb[r, c]]
            sn = _snap_to_catalog(det)
            if len(sn) != n:
                continue
            rn_, sn_ = _normalize(det), _normalize(sn)
            jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
            if jac < 0.70:
                continue
            err = abs(pitch - 58.0 * sc) / 58.0
            s = (ms + 0.050 * sup + 0.040 * n + 0.100 * run - 0.080 * eib
                 - 1.0 * err - 0.15 * (5 - n) + 0.20 * jac)
            if s > best_score:
                best_score = s
                pts = np.zeros((r1 - r0, c1 - c0, 2), dtype=int)
                for a in range(r0, r1):
                    for q in range(c0, c1):
                        pts[a - r0, q - c0] = (int(xc[i, q]), int(yc[j, a]))
                best = (pitch, float(dxa[i]), float(dya[j]), bb.copy(),
                        bv.copy(), pts, (r0, r1, c0, c1), s)
    if best is None:
        return None
    pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s = best
    rescued = []
    comp = _complete_winner_with_diamonds(
        sig, W, H, sx, sy, pitch, dx, dy, r0, c0, bb, slot_index, maps)
    if comp is not None:
        bb, r0, c0, rescued = comp[0], comp[1], comp[2], comp[3]
        bh, bw = bb.shape
        xi = np.round(sx + dx + cc * pitch).astype(int)
        yi = np.round(sy + dy + rr * pitch).astype(int)
        pts = np.zeros((bh, bw, 2), dtype=int)
        bv = np.zeros((bh, bw))
        for r in range(bh):
            for c in range(bw):
                pts[r, c] = (int(np.clip(xi[c0 + c], 0, W - 1)),
                             int(np.clip(yi[r0 + r], 0, H - 1)))
                bv[r, c] = float(sig[pts[r, c, 1], pts[r, c, 0]])
    return (pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s, rescued)


def draw_overlay(frame, board, pieces, sig, maps, W, H, sc, fps, nframe):
    vis = frame.copy()
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    rows, cols = board.rows, board.cols
    px = (gx2 - gx1) / max(cols, 1)
    py = (gy2 - gy1) / max(rows, 1)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 255, 255), 3)
    for i in range(cols + 1):
        x = int(gx1 + i * px)
        cv2.line(vis, (x, gy1), (x, gy2), (180, 180, 180), 1)
    for i in range(rows + 1):
        y = int(gy1 + i * py)
        cv2.line(vis, (gx1, y), (gx2, y), (180, 180, 180), 1)
    for r in range(rows):
        for c in range(cols):
            cx = int(gx1 + (c + 0.5) * px)
            cy = int(gy1 + (r + 0.5) * py)
            if board.grid[r, c]:
                cv2.circle(vis, (cx, cy), 14, (0, 255, 0), -1)
            else:
                cv2.circle(vis, (cx, cy), 6, (100, 100, 100), -1)
    half = config.PIECE_SLOT_HALF
    lines = [f"OBS frame={nframe} FPS={fps:.1f}",
             f"Grade: {rows}x{cols} fill={int(board.grid.sum())}"]
    for slot in range(3):
        sxn, syn = config.TRAY_SLOT_CENTERS[slot]
        sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
        cv2.rectangle(vis, (int(sxn - half), int(syn - half)),
                      (int(sxn + half), int(syn + half)), (255, 160, 0), 2)
        w = mirror_winner(sig, W, H, sx, sy, sc, maps, slot)
        p = pieces[slot] if slot < len(pieces) else None
        if w is None:
            lines.append(f"  S{slot}: sem vencedor")
            continue
        pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s, rescued = w
        bh, bw = bb.shape
        rset = set(tuple(x) for x in rescued)
        for r in range(bh):
            for c in range(bw):
                px_, py_ = int(pts[r, c, 0]), int(pts[r, c, 1])
                if (r, c) in rset:
                    cv2.circle(vis, (px_, py_), 18, (255, 0, 255), 3)
                    cv2.putText(vis, "DIAMANTE", (px_ + 20, py_ - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
                elif bb[r, c]:
                    cv2.circle(vis, (px_, py_), 14, (0, 255, 0), 2)
                else:
                    cv2.circle(vis, (px_, py_), 8, (0, 0, 255), 2)
        n = int(bb.sum())
        tag = f" +rescue{sorted(rset)}" if rset else ""
        lines.append(f"  S{slot}: {bw}x{bh} [{n}b] p={pitch:.0f} "
                     f"dx={dx:+.0f} dy={dy:+.0f} s={s:.2f}{tag}")
        # confere com o detector real
        real = sorted(p.cells) if p and p.cells else None
        mine = sorted((r, c) for r in range(bh) for c in range(bw) if bb[r, c])
        if real != mine:
            lines.append(f"    !! espelho!=real: real={real}")
    y = 30
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return vis


def main():
    print("== observador ao vivo ==", flush=True)
    # pre-voo
    r = subprocess.run([ADB, "devices"], capture_output=True, text=True, timeout=10)
    if DEVICE not in r.stdout:
        print("ABORTADO: dispositivo ausente", flush=True)
        return
    if user32.FindWindowW(None, "BlockBlaster Bot"):
        print("ABORTADO: bot ja em execucao", flush=True)
        return
    titles = []
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            b = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, b, 256)
            if b.value and "scrcpy" in b.value.lower():
                titles.append(b.value)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(cb), 0)
    if titles:
        print(f"ABORTADO: scrcpy estranho aberto {titles}", flush=True)
        return

    bot = subprocess.Popen([sys.executable, "-u", "bot.py"], cwd=PROJECT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("bot iniciado; aguardando janelas...", flush=True)
    hs = find_win(f"scrcpy_bot_{config.ADB_DEVICE_ID}")
    hb = find_win("BlockBlaster Bot")
    if not hs or not hb:
        print("ABORTADO: janelas nao apareceram", flush=True)
        bot.terminate()
        return
    # posiciona: scrcpy fica onde esta; bot vai p/ direita; overlay a esquerda
    sc = client_rect(hs)
    sw = user32.GetSystemMetrics(0)
    rb = win_rect(hb)
    user32.SetWindowPos(hb, 0, sw - rb[2] - 8, 0, 0, 0, 0x0001 | 0x0004 | 0x0010)
    time.sleep(0.5)
    sc = client_rect(hs)
    rb = win_rect(hb)
    ov_w = sc[2]
    ox = max(8, sc[0] - ov_w - 8)
    print(f"scrcpy cliente={sc} bot={rb} overlap={overlap(tuple(sc), tuple(rb))}px", flush=True)
    if overlap(tuple(sc), tuple(rb)) != 0:
        print("ABORTADO: overlap bot x scrcpy", flush=True)
        bot.terminate()
        return

    cv2.namedWindow("Observador", cv2.WINDOW_NORMAL)
    user32.SetWindowPos(user32.FindWindowW(None, "Observador"), 0, ox, sc[1],
                        0, 0, 0x0001 | 0x0004 | 0x0010)
    print(f"overlay em ({ox},{sc[1]}) tamanhocliente={sc[2]}x{sc[3]}", flush=True)

    t0 = time.time()
    nframe = 0
    last_save = 0.0
    t_prev = t0
    fps_show = 0.0
    target_sz = None
    try:
        with mss.mss() as sct:
            while time.time() - t0 < DURATION:
                if bot.poll() is not None:
                    print("bot encerrou sozinho", flush=True)
                    break
                cur = client_rect(hs)
                if cur is None or abs(cur[2] - sc[2]) > 4 or abs(cur[3] - sc[3]) > 4:
                    print("scrcpy moveu/sumiu — parando", flush=True)
                    break
                mon = {"left": cur[0], "top": cur[1], "width": cur[2], "height": cur[3]}
                try:
                    img = sct.grab(mon)
                except Exception:
                    time.sleep(0.1)
                    continue
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                H, W = frame.shape[:2]
                # normaliza p/ nativo (igual ao bot) p/ deteccao
                nat = cv2.resize(frame, (NATIVE_DEVICE_W, 2340)) if (W, H) != (NATIVE_DEVICE_W, 2340) else frame
                Hn, Wn = nat.shape[:2]
                board = detect_board(nat)
                pieces = read_pieces(nat)
                sig = _build_signal_map(nat)
                maps = _rescue_maps(nat)
                now = time.time()
                fps_show = 1.0 / max(now - t_prev, 1e-3)
                t_prev = now
                vis = draw_overlay(nat, board, pieces, sig, maps, Wn, Hn, Wn / NATIVE_DEVICE_W,
                                   fps_show, nframe)
                if target_sz is None:
                    ow = ov_w
                    oh = int(vis.shape[0] * ow / vis.shape[1])
                    target_sz = (ow, oh)
                    cv2.resizeWindow("Observador", ow, oh)
                cv2.imshow("Observador", cv2.resize(vis, target_sz))
                nframe += 1
                if time.time() - last_save > 30:
                    last_save = time.time()
                    cv2.imwrite(os.path.join(OUTDIR, f"obs_{nframe:04d}.png"), vis)
                    print(f"[{nframe}] grade={board.rows}x{board.cols} "
                          f"fill={int(board.grid.sum())} "
                          f"pecas={[len(p.cells) if p and p.cells else 0 for p in pieces]} "
                          f"fps={fps_show:.1f}", flush=True)
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    print("observador encerrado pelo usuario (q)", flush=True)
                    break
    finally:
        cv2.destroyAllWindows()
        if bot.poll() is None:
            try:
                hb2 = user32.FindWindowW(None, "BlockBlaster Bot")
                if hb2:
                    user32.SetForegroundWindow(hb2)
                    time.sleep(0.6)
                    if user32.GetForegroundWindow() == hb2:
                        user32.keybd_event(0x51, 0, 0, 0)
                        time.sleep(0.1)
                        user32.keybd_event(0x51, 0, 2, 0)
                        try:
                            bot.wait(timeout=15)
                            print("bot saiu graciosamente", flush=True)
                        except subprocess.TimeoutExpired:
                            bot.terminate()
                    else:
                        bot.terminate()
                else:
                    bot.terminate()
            except Exception:
                try:
                    bot.terminate()
                except Exception:
                    pass
        subprocess.run([ADB, "-s", DEVICE, "shell", "pkill", "-f", "monkey"],
                       capture_output=True, timeout=10)
        try:
            subprocess.run(["taskkill", "/F", "/IM", "scrcpy.exe"],
                           capture_output=True, timeout=10)
        except Exception:
            pass
        subprocess.run([ADB, "forward", "--remove", "tcp:1080"],
                       capture_output=True, timeout=10)
    print(f"fim. frames overlay: {nframe}. imagens em {OUTDIR}", flush=True)


def find_win(title, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.3)
    return 0


if __name__ == "__main__":
    main()
