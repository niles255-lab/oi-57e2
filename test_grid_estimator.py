#!/usr/bin/env python3
"""
Diagnóstico isolado do estimador de N (detect_grid_size).
NÃO altera vision.py/bot.py. Apenas mede e salva visualizações.
Sem jogadas reais.
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import adb_io
from vision import _score_grid_n_grad

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_grid")
os.makedirs(OUT, exist_ok=True)


def projections(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    abs_gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    col_proj = abs_gx.mean(axis=0)
    row_proj = abs_gy.mean(axis=1)
    col_proj = cv2.GaussianBlur(col_proj.reshape(1, -1), (1, 7), 1.5).reshape(-1)
    row_proj = cv2.GaussianBlur(row_proj.reshape(-1, 1), (7, 1), 1.5).reshape(-1)
    return col_proj, row_proj


def score_table(proj, name):
    print(f"--- scores {name} ---", flush=True)
    tab = {}
    for n in (5, 6, 7, 8, 9):
        s = _score_grid_n_grad(proj, n)
        tab[n] = s
        print(f"  N={n}: score={s:.4f}", flush=True)
    best = max(tab, key=tab.get)
    vals = sorted(tab.values(), reverse=True)
    conf = (vals[0] - vals[1]) / vals[0] if vals[0] > 0 else 0.0
    print(f"  vencedor={best} confianca={(conf*100):.1f}% (limiar 8%)", flush=True)
    return tab


def draw_separators(crop, n_cols, n_rows, path):
    h, w = crop.shape[:2]
    img = crop.copy()
    for i in range(n_cols + 1):
        x = int(i * w / n_cols)
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
    for i in range(n_rows + 1):
        y = int(0 + i * h / n_rows)
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
    cv2.imwrite(path, img)


def draw_profile(proj, n_list, path, horizontal=False):
    L = len(proj)
    H, W = 200, max(600, L)
    img = np.zeros((H, W, 3), np.uint8)
    mx = proj.max() if proj.max() > 0 else 1.0
    xs = np.linspace(0, W - 1, L).astype(int)
    ys = (H - 10 - (proj / mx) * (H - 20)).astype(int)
    for i in range(1, L):
        cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (200, 200, 200), 1)
    colors = {7: (0, 0, 255), 8: (0, 255, 0)}
    for n in n_list:
        pitch = L / n
        for i in range(1, n):
            p = int(i * pitch / L * (W - 1))
            cv2.line(img, (p, 0), (p, H), colors.get(n, (255, 255, 0)), 1)
    if horizontal:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(path, img)


def main():
    print("== diagnóstico do estimador de N ==", flush=True)
    ok = adb_io.init_scrcpy_stream()
    print(f"stream: {ok}", flush=True)
    time.sleep(2)

    fr = adb_io.capture_frame()
    print(f"frame: {tuple(fr.shape)}", flush=True)
    cv2.imwrite(os.path.join(OUT, "frame_native.png"), fr)

    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    crop = fr[gy1:gy2, gx1:gx2]
    print(f"crop GRID: shape={tuple(crop.shape)} (regiao {gx1},{gy1}-{gx2},{gy2})", flush=True)
    cv2.imwrite(os.path.join(OUT, "crop_grid.png"), crop)

    col_proj, row_proj = projections(crop)
    print(f"col_proj len={len(col_proj)} max={col_proj.max():.2f} | row_proj len={len(row_proj)} max={row_proj.max():.2f}", flush=True)
    score_table(col_proj, "colunas")
    score_table(row_proj, "linhas")

    draw_separators(crop, 7, 7, os.path.join(OUT, "crop_sep_7x7.png"))
    draw_separators(crop, 8, 8, os.path.join(OUT, "crop_sep_8x8.png"))
    draw_profile(col_proj, [7, 8], os.path.join(OUT, "proj_col.png"))
    draw_profile(row_proj, [7, 8], os.path.join(OUT, "proj_row.png"), horizontal=True)

    # Hipótese borda externa: recorta 4% de cada lado e reavalia
    h, w = crop.shape[:2]
    mx, my = int(w * 0.04), int(h * 0.04)
    inner = crop[my:h - my, mx:w - mx]
    print(f"crop interno (sem 4% bordas): shape={tuple(inner.shape)}", flush=True)
    cv2.imwrite(os.path.join(OUT, "crop_grid_inner.png"), inner)
    ci, ri = projections(inner)
    score_table(ci, "colunas (sem borda)")
    score_table(ri, "linhas (sem borda)")

    # Efeito do upscale: compara screencap nativo direto vs normalizado
    print("--- screencap nativo direto (sem upscale) ---", flush=True)
    res = adb_io._cmd(["exec-out", "screencap", "-p"], timeout=20.0)
    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    nat = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    print(f"screencap shape: {tuple(nat.shape)}", flush=True)
    crop_nat = nat[gy1:gy2, gx1:gx2]
    cn, rn = projections(crop_nat)
    score_table(cn, "colunas (screencap nativo)")
    score_table(rn, "linhas (screencap nativo)")

    adb_io.stop_scrcpy_stream()
    print(f"imagens em: {OUT}", flush=True)


if __name__ == "__main__":
    main()
