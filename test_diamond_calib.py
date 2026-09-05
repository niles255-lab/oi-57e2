#!/usr/bin/env python3
"""
Calibracao da assinatura do diamante no frame live (diagnostico apenas).
Compara: celula com diamante (S2) vs triangulo (S1) vs roxo normal vs vazio.
Nenhuma alteracao no codigo do jogo. Nenhuma jogada.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", "diag_readonly", "live_20260904_185226.png")

# Regioes aproximadas (x1,y1,x2,y2) medidas na imagem 1080x2340:
# S2: barra roxa y~1760-1820 x~560-730; diamante cinza acima-direita
REGIONS = {
    "S2_diamante": (640, 1680, 730, 1770),
    "S2_roxo": (560, 1760, 640, 1830),
    "S1_triangulo": (470, 1760, 540, 1830),
    "S1_roxo": (400, 1690, 470, 1760),
    "vazio_escuro": (100, 1900, 200, 2000),
}


def stats(name, img, box):
    x1, y1, x2, y2 = box
    sub = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)
    n = h.size
    gray = ((s < 65) & (v >= 110)).mean()
    pink = ((h >= 135) & (h <= 168) & (s > 80) & (v > 100)).mean()
    pink_hi = ((h >= 135) & (h <= 168) & (s > 80) & (v > 150)).mean()
    # centro (terco central)
    hh, ww = h.shape
    cx, cy = slice(ww // 3, 2 * ww // 3), slice(hh // 3, 2 * hh // 3)
    hc, sc, vc = h[cy, cx], s[cy, cx], v[cy, cx]
    pink_c = ((hc >= 135) & (hc <= 168) & (sc > 80) & (vc > 100)).mean()
    print(f"{name:14s} Smed={s.mean():6.1f} Vmed={v.mean():6.1f} "
          f"gray={gray:.3f} pink={pink:.3f} pinkHiV={pink_hi:.3f} pinkCentro={pink_c:.3f}",
          flush=True)
    return {"gray": gray, "pink": pink, "pink_c": pink_c}


def main():
    fr = cv2.imread(FRAME_PATH)
    assert fr is not None, FRAME_PATH
    print(f"frame: {fr.shape[1]}x{fr.shape[0]}", flush=True)
    for name, box in REGIONS.items():
        stats(name, fr, box)


if __name__ == "__main__":
    main()
