#!/usr/bin/env python3
"""
Memória/catálogo das geometrias das grades (tabuleiros NxN).

Somente leitura de frames + escrita em data/visual_memory/boards/.
NÃO altera detector, solver, score, thresholds, ADB ou autoplay.

Estrutura:
  data/visual_memory/boards/
    5x5/profile.json + grid_crop.png + full_frame.png + drawing.png
    ...
    11x11/...
"""

import os
import json
import datetime

import cv2
import numpy as np

from .vision import _score_grid_n_grad

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "visual_memory", "boards")
INDEX_PATH = os.path.join(BASE, "index.json")

CANDIDATES = (5, 6, 7, 8, 9, 10, 11, 12)
SQUARE_TOL = 0.03  # |w-h|/max permitido
CONF_MARGIN = 0.08  # melhor >= 8% acima do segundo

# Compatibilidade entre geometria detectada e perfil salvo
CORNER_TOL_PX = 8    # cantos dentro de 8 px
PITCH_TOL_REL = 0.03  # pitch dentro de 3%


def _cluster_lines(vals, tol=5):
    """Agrupa coordenadas (y de H ou x de V); retorna [(centro, força)].

    Força = soma dos comprimentos dos segmentos no grupo.
    """
    groups = []
    for v, length in sorted(vals):
        for g in groups:
            if abs(g[0] - v) <= tol:
                g[0] = (g[0] * g[1] + v * length) / (g[1] + length)
                g[1] += length
                break
        else:
            groups.append([float(v), float(length)])
    return sorted(((c, f) for c, f in groups), key=lambda x: -x[1])


def _hough_rects(gray, canny_lo, canny_hi, hough_thr, min_len):
    ed = cv2.Canny(gray, canny_lo, canny_hi)
    lines = cv2.HoughLinesP(ed, 1, np.pi / 180, threshold=hough_thr,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return []
    hor, ver = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(int):
        if abs(x2 - x1) > 2 * abs(y2 - y1):
            hor.append((min(y1, y2), max(x1, x2) - min(x1, x2)))
        elif abs(y2 - y1) > 2 * abs(x2 - x1):
            ver.append((min(x1, x2), max(y1, y2) - min(y1, y2)))
    hg = _cluster_lines([(y, ln) for y, ln in hor])[:3]
    vg = _cluster_lines([(x, ln) for x, ln in ver])[:3]
    if len(hg) < 2 or len(vg) < 2:
        return []
    H, W = gray.shape[:2]
    out = []
    for (ya, fa) in hg:
        for (yb, fb) in hg:
            if yb - ya < 100:
                continue
            for (xa, ga) in vg:
                for (xb, gb) in vg:
                    if xb - xa < 100:
                        continue
                    w, h = xb - xa, yb - ya
                    if abs(w - h) / max(w, h) > SQUARE_TOL:
                        continue  # nunca aceitar retangular
                    out.append(((fa + fb + ga + gb, w * h),
                                int(round(xa)), int(round(ya)),
                                int(round(xb)), int(round(yb))))
    return out


def find_outer_rect(frame):
    """Borda externa da grade (linhas longas de Hough). Retorna
    (x1, y1, x2, y2) ou None (nunca inventa coordenadas).

    Tenta parâmetros estritos primeiro (frames ADB nítidos) e cai para
    sensíveis (frames de stream comprimidos/upscale). Escolhe o retângulo
    quadrado de maior força; empate, maior área.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    for params in ((60, 180, 300, 500), (40, 120, 150, 300)):
        best = None
        for key, x1, y1, x2, y2 in _hough_rects(gray, *params):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 100 or y2 - y1 < 100:
                continue
            if best is None or key > best[0]:
                best = (key, x1, y1, x2, y2)
        if best is not None:
            _, x1, y1, x2, y2 = best
            return (x1, y1, x2, y2)
    return None


def _proj_scores(proj, size):
    return {n: _score_grid_n_grad(proj, n) for n in CANDIDATES}


def _confident(scores):
    vals = sorted(scores.values(), reverse=True)
    return vals[0] > 0 and (vals[0] - vals[1]) / vals[0] >= CONF_MARGIN


def analyze_board(frame):
    """Analisa UMA frame (sem salvar). Retorna dict do perfil ou None.

    Inclui: N, cantos, pitches, centros, scores/margens, quadrado ok,
    confiante (bool). Nunca inventa coordenadas: sem retângulo externo
    ou sem margem de confiança, retorna None.
    """
    H, W = frame.shape[:2]
    rect = find_outer_rect(frame)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    if abs(w - h) / max(w, h) > SQUARE_TOL:
        return None  # nunca aceitar retangular como válido
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    crop = gray[y1:y2, x1:x2]
    gx = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    gy = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    gx = cv2.GaussianBlur(gx.reshape(1, -1), (1, 7), 1.5).reshape(-1)
    gy = cv2.GaussianBlur(gy.reshape(-1, 1), (7, 1), 1.5).reshape(-1)
    cs = _proj_scores(gx, w)
    rs = _proj_scores(gy, h)
    nc = max(cs, key=cs.get)
    nr = max(rs, key=rs.get)
    if nr != nc or not _confident(cs) or not _confident(rs):
        return None
    n = nc
    pitchx, pitchy = w / n, h / n
    centers = [[round(x1 + (c + 0.5) * pitchx, 2),
                round(y1 + (r + 0.5) * pitchy, 2)]
               for r in range(n) for c in range(n)]
    vc, wc = sorted(cs.values(), reverse=True)[:2]
    vr, wr = sorted(rs.values(), reverse=True)[:2]
    return {
        "grid_size": n,
        "shape": f"{n}x{n}",
        "square": True,
        "frame_size": [W, H],
        "grid_top_left": [x1, y1],
        "grid_bottom_right": [x2, y2],
        "area_px": [w, h],
        "pitch_x": round(float(pitchx), 3),
        "pitch_y": round(float(pitchy), 3),
        "cell_centers_xy": centers,
        "offsets": {"dx": 0.0, "dy": 0.0},
        "col_score": round(vc, 1),
        "row_score": round(vr, 1),
        "col_margin_pct": round(100 * (vc - wc) / vc, 1),
        "row_margin_pct": round(100 * (vr - wr) / vr, 1),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def profile_dir(n):
    return os.path.join(BASE, f"{n}x{n}")


def _load_index():
    if not os.path.exists(INDEX_PATH):
        return {"boards": {}}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            idx = json.load(f)
        if not isinstance(idx.get("boards"), dict):
            return {"boards": {}}
        return idx
    except (OSError, ValueError):
        return {"boards": {}}


def _save_index(idx):
    os.makedirs(BASE, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def register_profile(profile):
    """Registra perfil no índice (sem apagar os demais). Retorna o índice."""
    idx = _load_index()
    n = profile["grid_size"]
    idx["boards"][str(n)] = {
        "shape": profile.get("shape", f"{n}x{n}"),
        "profile": os.path.join(f"{n}x{n}", "profile.json"),
        "grid_top_left": profile.get("grid_top_left"),
        "grid_bottom_right": profile.get("grid_bottom_right"),
        "pitch_x": profile.get("pitch_x"),
        "pitch_y": profile.get("pitch_y"),
        "timestamp": profile.get("timestamp"),
    }
    _save_index(idx)
    return idx


def load_profile(n):
    p = os.path.join(profile_dir(n), "profile.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_profiles():
    out = []
    idx = _load_index()
    if idx["boards"]:
        for key in sorted(idx["boards"], key=int):
            p = load_profile(int(key))
            if p is not None:
                out.append(p)
        if out:
            return out
    if not os.path.isdir(BASE):
        return out
    for name in sorted(os.listdir(BASE)):
        p = os.path.join(BASE, name, "profile.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                continue
    return out


def save_profile(profile, frame):
    """Grava perfil + PNGs. Retorna (dir, created: bool). Nunca apaga outros."""
    n = profile["grid_size"]
    d = profile_dir(n)
    created = not os.path.exists(os.path.join(d, "profile.json"))
    os.makedirs(d, exist_ok=True)
    x1, y1 = profile["grid_top_left"]
    x2, y2 = profile["grid_bottom_right"]
    cv2.imwrite(os.path.join(d, "grid_crop.png"), frame[y1:y2, x1:x2])
    cv2.imwrite(os.path.join(d, "full_frame.png"), frame)
    ov = frame.copy()
    for i in range(n + 1):
        x = int(round(x1 + i * profile["pitch_x"]))
        y = int(round(y1 + i * profile["pitch_y"]))
        cv2.line(ov, (x, y1), (x, y2), (0, 255, 0), 2)
        cv2.line(ov, (x1, y), (x2, y), (0, 255, 0), 2)
    for (x, y) in profile["cell_centers_xy"]:
        cv2.circle(ov, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(d, "drawing.png"), ov)
    prof = dict(profile)
    prof["files"] = ["full_frame.png", "grid_crop.png", "drawing.png"]
    with open(os.path.join(d, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)
    register_profile(prof)
    return d, created


def geometry_compatible(detected, saved):
    """True se a geometria detectada bate com o perfil salvo.

    Compara cantos (px) e pitch (relativo). Mesma dimensão com geometria
    divergente retorna False (variação: pausar, nunca sobrescrever).
    """
    try:
        dtl, dbr = detected["grid_top_left"], detected["grid_bottom_right"]
        stl, sbr = saved["grid_top_left"], saved["grid_bottom_right"]
        if (abs(dtl[0] - stl[0]) > CORNER_TOL_PX
                or abs(dtl[1] - stl[1]) > CORNER_TOL_PX
                or abs(dbr[0] - sbr[0]) > CORNER_TOL_PX
                or abs(dbr[1] - sbr[1]) > CORNER_TOL_PX):
            return False
        for k in ("pitch_x", "pitch_y"):
            ref = float(saved[k])
            if abs(float(detected[k]) - ref) / ref > PITCH_TOL_REL:
                return False
        return True
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _profile_confident(frame, saved):
    """Via rápida: o perfil salvo ainda vale nesta frame?

    Pontua os separadores do N salvo DENTRO da região salva. Robusto a
    blur de stream (não precisa da borda externa)."""
    try:
        n = int(saved["grid_size"])
        x1, y1 = saved["grid_top_left"]
        x2, y2 = saved["grid_bottom_right"]
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 100 or y2 - y1 < 100:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop = gray[y1:y2, x1:x2]
        gx = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        gy = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        gx = cv2.GaussianBlur(gx.reshape(1, -1), (1, 7), 1.5).reshape(-1)
        gy = cv2.GaussianBlur(gy.reshape(-1, 1), (7, 1), 1.5).reshape(-1)
        cs = {m: _score_grid_n_grad(gx, m) for m in CANDIDATES}
        rs = {m: _score_grid_n_grad(gy, m) for m in CANDIDATES}
        if max(cs, key=cs.get) != n or max(rs, key=rs.get) != n:
            return False
        return _confident(cs) and _confident(rs)
    except (KeyError, TypeError, ValueError):
        return False


def ensure_board(frame):
    """Porta de reconhecimento para o autoplay (somente leitura).

    Retorna (status, detected, saved) onde status é:
      'known'   — perfil catalogado confere nesta frame (seguir sem perguntar)
      'new'     — dimensão desconhecida (pausar antes de qualquer jogada)
      'variant' — mesma dimensão, geometria divergente (pausar, não sobrescrever)
      'unknown' — grade não detectada com confiança (pausar; nada salvo)
    Nunca escreve nada e nunca pede confirmação (o chamador decide).
    """
    detected = analyze_board(frame)
    if detected is None:
        # Sem borda confiável: tenta os perfis salvos na região deles
        # (vale para frames de stream, onde a borda externa some no blur).
        for saved in list_profiles():
            try:
                n = int(saved.get("grid_size", 0))
            except (TypeError, ValueError):
                continue
            if _profile_confident(frame, saved):
                return ("known", {"grid_size": n, "shape": f"{n}x{n}"}, saved)
        return ("unknown", None, None)
    saved = load_profile(detected["grid_size"])
    if saved is None:
        return ("new", detected, None)
    if geometry_compatible(detected, saved):
        return ("known", detected, saved)
    return ("variant", detected, saved)


def ascii_map(n):
    return "\n".join([" ".join(["."] * n) for _ in range(n)])
