#!/usr/bin/env python3
"""
Catalogo visual de pecas (somente leitura + escrita em data/).
NAO altera detector, solver, autoplay ou execucao. Nao joga.

Fluxo: escanear -> preview -> (confirmacao no chat) -> salvar.
"""

import os
import json
import datetime
import cv2
import numpy as np

from . import config
from .vision import (read_pieces, _block_mask, _build_signal_map,
                    _rescue_maps, _snap_to_catalog, _normalize, _max_run,
                    _complete_winner_with_diamonds, score_candidate,
                    _is_native_scale,
                    NATIVE_DEVICE_W, NATIVE_DEVICE_H)
from .visual_memory import detect_diamond_cells


def _expand_pts(sig, W, H, sx, sy, pitch, dx, dy, r0, c0, bh, bw):
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    xi = np.round(sx + dx + cc * pitch).astype(int)
    yi = np.round(sy + dy + rr * pitch).astype(int)
    pts = np.zeros((bh, bw, 2), dtype=int)
    bv = np.zeros((bh, bw))
    for r in range(bh):
        for c in range(bw):
            pts[r, c] = (int(np.clip(xi[c0 + c], 0, W - 1)),
                         int(np.clip(yi[r0 + r], 0, H - 1)))
            bv[r, c] = float(sig[pts[r, c, 1], pts[r, c, 0]])
    return pts, bv


def mirror_winner(sig, W, H, sx, sy, sc, maps, slot_index=0):
    """Replica fiel da busca atual do detect_piece (diagnostico).
    Retorna (pitch, dx, dy, bb, bv, pts, (r0,r1,c0,c1), score, rescued)
    ou None. rescue[] vazio quando nada resgatado."""
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
            n_max = 9 if _is_native_scale(sc, H / NATIVE_DEVICE_H) else 5
            if n < 1 or n > n_max:
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
            if sn is None or len(sn) != n:
                continue
            rn_, sn_ = _normalize(det), _normalize(sn)
            jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
            if jac < 0.70:
                continue
            s = score_candidate(ms, sup, n, run, eib, pitch, sc,
                                H / NATIVE_DEVICE_H,
                                int((bv[bb] < 0.75).sum()), jac)
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
        r1, c1 = r0 + bh, c0 + bw
        pts, bv = _expand_pts(sig, W, H, sx, sy, pitch, dx, dy, r0, c0, bh, bw)
    return (pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s, rescued)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_ROOT = os.path.join(BASE, "data", "visual_memory", "pieces", "catalog")
NORMAL_DIR = os.path.join(CATALOG_ROOT, "normal")
DIAMOND_DIR = os.path.join(CATALOG_ROOT, "diamond")
INDEX_PATH = os.path.join(CATALOG_ROOT, "index.json")


def normalize_cells(cells):
    """Bounding box minima com origem (0,0). Posicao absoluta irrelevante."""
    cells = [(int(r), int(c)) for r, c in cells]
    if not cells:
        return []
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return sorted((r - mr, c - mc) for r, c in cells)


def shape_key(cells):
    return tuple(normalize_cells(cells))


def dims_of(cells):
    n = normalize_cells(cells)
    if not n:
        return (0, 0)
    return (max(r for r, _ in n) + 1, max(c for _, c in n) + 1)


def _dia_set(diamonds):
    """Diamantes ja vindo no referencial normalizado da forma: so converte
    para tupla de int, SEM deslocar (deslocar aqui recriaria o bug que
    transformava [(0,2)] em [(0,0)])."""
    if not diamonds:
        return set()
    return set((int(r), int(c)) for r, c in diamonds)


def _load_index():
    if not os.path.exists(INDEX_PATH):
        return {"next_id": 1, "records": []}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(idx):
    os.makedirs(CATALOG_ROOT, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def _norm_pair(shape, diamonds):
    """Normaliza forma e diamantes COM O MESMO deslocamento.

    O diamante pertence a geometria da peca: normalizar o subconjunto
    sozinho deslocaria a coordenada (bug: [(0,2)] virava [(0,0)]).
    """
    cells = sorted((int(r), int(c)) for r, c in shape)
    if not cells:
        return [], []
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    norm = sorted((r - mr, c - mc) for r, c in cells)
    dia = sorted((int(r) - mr, int(c) - mc) for r, c in (diamonds or []))
    return norm, dia


def find_match(shape, diamonds):
    """Compara forma+diamantes com o catalogo. Retorna (record|None, motivo)."""
    idx = _load_index()
    key = shape_key(shape)
    _, dia = _norm_pair(shape, diamonds)
    dia = [list(d) for d in dia]
    same_shape = [r for r in idx["records"]
                  if tuple(map(tuple, r["shape"])) == tuple(key)]
    if not same_shape:
        return None, "forma nova"
    for r in same_shape:
        if sorted(r.get("diamond_cells", [])) == dia:
            return r, f"duplicata de {r['id']}"
    kinds = sorted(set(r["kind"] for r in same_shape))
    return None, (f"forma conhecida ({', '.join(kinds)}), "
                  f"mas configuracao de diamante nova: {dia}")


def dominant_color_bgr(frame, centers_px, half_px=8):
    """Cor dominante amostrada no centro dos blocos (para o drawing)."""
    H, W = frame.shape[:2]
    acc = np.zeros(3, dtype=np.float64)
    n = 0
    for (cx, cy) in centers_px:
        x1, x2 = max(0, cx - half_px), min(W, cx + half_px + 1)
        y1, y2 = max(0, cy - half_px), min(H, cy + half_px + 1)
        if x2 > x1 and y2 > y1:
            acc += frame[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
            n += 1
    if n == 0:
        return (200, 200, 200)
    return tuple(int(v) for v in (acc / n))


def emoji_for_bgr(bgr):
    b, g, r = [int(v) for v in bgr]
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0])
    if h < 10 or h >= 155:
        return "🟥"
    if h < 25:
        return "🟧"
    if h < 40:
        return "🟨"
    if h < 85:
        return "🟩"
    if h < 130:
        return "🟦"
    return "🟪"


def emoji_drawing(shape, diamonds, color_bgr):
    """Desenho em emoji preservando a forma (para o chat)."""
    cells = normalize_cells(shape)
    dia = _dia_set(diamonds)
    if not cells:
        return "(vazia)"
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    base = emoji_for_bgr(color_bgr)
    grid = [["⬛" for _ in range(w)] for _ in range(h)]
    for (r, c) in cells:
        grid[r][c] = "💎" if (r, c) in dia else base
    return "\n".join(" ".join(row) for row in grid)


def ascii_drawing(shape, diamonds):
    cells = normalize_cells(shape)
    dia = _dia_set(diamonds)
    if not cells:
        return "(vazia)"
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    grid = [["." for _ in range(w)] for _ in range(h)]
    for (r, c) in cells:
        grid[r][c] = "D" if (r, c) in dia else "#"
    return "\n".join("".join(row) for row in grid)


def draw_piece_png(shape, diamonds, color_bgr, path, cell_px=64):
    """drawing.png canonico: blocos na cor da peca, diamante diferenciado."""
    cells = normalize_cells(shape)
    dia = _dia_set(diamonds)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    m = 20
    img = np.zeros((m * 2 + h * cell_px, m * 2 + w * cell_px, 3), np.uint8)
    img[:] = (18, 18, 18)
    b, g, r = [int(v) for v in color_bgr]
    for (rr, cc) in cells:
        x1, y1 = m + cc * cell_px + 3, m + rr * cell_px + 3
        x2, y2 = m + (cc + 1) * cell_px - 3, m + (rr + 1) * cell_px - 3
        if (rr, cc) in dia:
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            d = cell_px // 4
            pts = np.array([[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]])
            cv2.fillPoly(img, [pts], (255, 0, 255))
            cv2.polylines(img, [pts], True, (255, 255, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (max(0, b - 70), max(0, g - 70), max(0, r - 70)), 3)
            cv2.line(img, (x1 + 6, y1 + 6), (x2 - 6, y1 + 6), (255, 255, 255), 2)
    cv2.imwrite(path, img)


def analyze_slot(frame, slot):
    """Analisa UM slot: deteccao real + geometria do espelho + diamantes.

    Retorna dict ou None (slot vazio). Nao escreve nada em disco.
    """
    H, W = frame.shape[:2]
    sc = W / NATIVE_DEVICE_W
    pieces = read_pieces(frame)
    p = pieces[slot] if slot < len(pieces) else None
    if p is None or not p.cells:
        return None
    sig = _build_signal_map(frame)
    maps = _rescue_maps(frame)
    sxn, syn = config.TRAY_SLOT_CENTERS[slot]
    sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
    w = mirror_winner(sig, W, H, sx, sy, sc, maps, slot)
    if w is None:
        return None
    pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s, rescued = w
    bh, bw = bb.shape
    # so prossegue se o espelho concordar com o detector real
    if sorted(p.cells) != sorted((r, c) for r in range(bh) for c in range(bw) if bb[r, c]):
        return {"mismatch": True, "piece": p, "pitch": pitch,
                "dx": dx, "dy": dy, "score": s}
    # centros dos blocos em coordenadas do frame -> diamante por celula
    half_slot = int(config.PIECE_SLOT_HALF * sc)
    cx0 = int(max(0, sx - half_slot))
    cy0 = int(max(0, sy - half_slot))
    crop = frame[cy0:int(min(H, sy + half_slot)), cx0:int(min(W, sx + half_slot))]
    cells_px, order = [], []
    for r in range(bh):
        for c in range(bw):
            if bb[r, c]:
                order.append((r, c))
                cells_px.append((int(pts[r, c, 0]) - cx0, int(pts[r, c, 1]) - cy0))
    from .visual_memory import detect_diamond_cells
    dia = detect_diamond_cells(crop, cells_px, max(4, int(pitch / 2)))
    diamond_cells = sorted([order[int(k)] for k, v in dia.items() if v.get("has_diamond")])
    diamond_classes = {order[int(k)]: v.get("class") for k, v in dia.items()
                       if v.get("has_diamond")}
    centers = [(int(pts[r, c, 0]), int(pts[r, c, 1]))
               for r in range(bh) for c in range(bw) if bb[r, c]]
    color = dominant_color_bgr(frame, centers)
    return {"piece": p, "shape": sorted(p.cells),
            "diamond_cells": diamond_cells, "diamond_classes": diamond_classes,
            "cells_px": cells_px, "cell_order": order,
            "diamond_half_px": max(4, int(pitch / 2)),
            "pitch": pitch, "dx": dx, "dy": dy,
            "score": s, "color": color, "crop": crop,
            "mask": None, "frame_size": (W, H), "scale": sc}


def describe_diamonds(a):
    """Aparência de cada diamante (somente leitura; tolera análises sem cells_px).

    Retorna lista [{cell, class, appearance}] na ordem das diamond_cells.
    """
    if a is None or not a.get("diamond_cells"):
        return []
    from .visual_memory import diamond_cell_appearance
    cells_px = a.get("cells_px") or []
    order = a.get("cell_order") or []
    half = int(a.get("diamond_half_px") or 8)
    pos = {tuple(order[i]): cells_px[i] for i in range(min(len(order), len(cells_px)))}
    out = []
    for cell in a["diamond_cells"]:
        key = tuple(cell)
        cls = (a.get("diamond_classes") or {}).get(key)
        ap = None
        if key in pos:
            ap = diamond_cell_appearance(a.get("crop"), pos[key][0],
                                         pos[key][1], half, cls)
        out.append({"cell": list(cell), "class": cls, "appearance": ap})
    return out


def diamond_type_status(cls_name):
    """(conhecido?, observações) do tipo visual no registro de tipos."""
    if not cls_name:
        return False, 0
        from .visual_memory import diamond_type_index
    entry = diamond_type_index().get("types", {}).get(cls_name)
    if not entry:
        return False, 0
    return True, int(entry.get("count", 0))


def finalize_masks(analysis):
    """Preenche a mascara do slot (para salvar junto)."""
    if analysis is None or "crop" not in analysis:
        return analysis
    import cv2 as _cv
    hsv = _cv.cvtColor(analysis["crop"], _cv.COLOR_BGR2HSV)
    analysis["mask"] = _block_mask(hsv)
    return analysis


def preview_text(slot, a):
    """Bloco de confirmacao (para mostrar antes de salvar)."""
    if a is None:
        return f"SLOT {slot}\nPeça ausente (slot vazio)."
    if a.get("mismatch"):
        p = a["piece"]
        return (f"SLOT {slot}\nEspelho divergiu do detector "
                f"({sorted(p.cells)}); analise abortada p/ este slot.")
    lines = [f"SLOT {slot}", "", "FORMA DETECTADA:", ascii_drawing(a["shape"], a["diamond_cells"]),
             "", "DESENHO:", emoji_drawing(a["shape"], a["diamond_cells"], a["color"]), "",
             f"BLOCOS: {len(a['shape'])}",
             f"DIMENSÃO: {a['piece'].height}x{a['piece'].width}",
             f"DIAMANTES: {len(a['diamond_cells'])}" +
             (f" em {a['diamond_cells']}" if a["diamond_cells"] else " (nenhum)"),
             f"pitch={a['pitch']:.1f} dx={a['dx']:+.0f} dy={a['dy']:+.0f} "
             f"conf={a['score']:.3f} slot_origem={slot}"]
    return "\n".join(lines)


def preview_text_ascii(slot, a):
    """Versao ASCII do preview (fallback p/ terminal Windows cp1252).

    Mesmo conteudo de preview_text(), mas DESENHO usa ascii_drawing()
    (sem emoji). Nao altera deteccao; somente apresentacao.
    """
    if a is None:
        return f"SLOT {slot}\nPeca ausente (slot vazio)."
    if a.get("mismatch"):
        p = a["piece"]
        return (f"SLOT {slot}\nEspelho divergiu do detector "
                f"({sorted(p.cells)}); analise abortada p/ este slot.")
    lines = [f"SLOT {slot}", "", "FORMA DETECTADA:", ascii_drawing(a["shape"], a["diamond_cells"]),
             "", "DESENHO (ASCII):", ascii_drawing(a["shape"], a["diamond_cells"]), "",
             f"BLOCOS: {len(a['shape'])}",
             f"DIMENSAO: {a['piece'].height}x{a['piece'].width}",
             f"DIAMANTES: {len(a['diamond_cells'])}" +
             (f" em {a['diamond_cells']}" if a["diamond_cells"] else " (nenhum)"),
             f"pitch={a['pitch']:.1f} dx={a['dx']:+.0f} dy={a['dy']:+.0f} "
             f"conf={a['score']:.3f} slot_origem={slot}"]
    return "\n".join(lines)


def _color_for_record(kind, rid, default=(132, 53, 83)):
    """Cor BGR p/ desenho emoji no status (somente leitura).

    Tenta ler dominant_color_bgr do metadata.json do registro;
    se ausente, usa roxo padrao (mapeia p/ 🟪 em emoji_for_bgr).
    """
    try:
        meta_path = os.path.join(
            NORMAL_DIR if kind == "normal" else DIAMOND_DIR, rid, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        col = meta.get("dominant_color_bgr")
        if isinstance(col, (list, tuple)) and len(col) == 3:
            return tuple(int(v) for v in col)
    except (OSError, ValueError):
        pass
    return default


def catalog_summary():
    """Resumo somente-leitura do catalogo (p/ --list/--status).

    Retorna dict com contagens e registros enriquecidos com rows/cols.
    Nao captura tela, nao usa ADB, nao executa solver.
    """
    idx = _load_index()
    records = idx.get("records", [])
    normals = sum(1 for r in records if r.get("kind") == "normal")
    diamonds = sum(1 for r in records if r.get("kind") == "diamond")
    enriched = []
    for r in records:
        shape = [tuple(c) for c in r.get("shape", [])]
        rows, cols = dims_of(shape)
        enriched.append({
            "id": r.get("id"),
            "kind": r.get("kind"),
            "full_id": f"{r.get('kind')}/{r.get('id')}",
            "shape": r.get("shape", []),
            "diamond_cells": r.get("diamond_cells", []),
            "block_count": r.get("block_count", len(shape)),
            "rows": rows,
            "cols": cols,
        })
    return {"normals": normals, "diamonds": diamonds,
            "total": len(records), "records": enriched,
            "next_id": idx.get("next_id")}


def status_text(use_emoji=True):
    """Texto do CATALOGO OK (somente leitura; nao altera deteccao).

    use_emoji=True usa emoji_drawing (pode falhar em cp1252; o chamador
    deve fazer fallback p/ use_emoji=False). use_emoji=False usa
    ascii_drawing e nunca levanta UnicodeEncodeError por emoji.
    """
    s = catalog_summary()
    lines = ["CATALOGO OK", "", f"Normais: {s['normals']}",
             f"Com diamante: {s['diamonds']}", f"Total: {s['total']}", ""]
    if not s["records"]:
        lines.append("(catalogo vazio)")
        return "\n".join(lines)
    for r in s["records"]:
        shape = [tuple(c) for c in r["shape"]]
        dia = [tuple(c) for c in r["diamond_cells"]]
        lines.append(f"- ID: {r['full_id']}")
        lines.append(f"  tipo: {r['kind']}")
        lines.append(f"  blocos: {r['block_count']}")
        lines.append(f"  dimensao: {r['rows']}x{r['cols']}")
        lines.append(f"  diamond_cells: {r['diamond_cells']}")
        lines.append("  desenho:")
        if use_emoji:
            color = _color_for_record(r["kind"], r["id"])
            lines.append(emoji_drawing(shape, dia, color))
        else:
            lines.append(ascii_drawing(shape, dia))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_record(analysis, slot):
    """Salva registro (normal|diamond) + PNGs + metadata.json. Retorna (id, dir)."""
    a = finalize_masks(analysis)
    shape, dia = _norm_pair(a["shape"], a["diamond_cells"])
    kind = "diamond" if dia else "normal"
    idx = _load_index()
    rid = f"piece_{idx['next_id']:03d}"
    idx["next_id"] += 1
    d = os.path.join(NORMAL_DIR if kind == "normal" else DIAMOND_DIR, rid)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "original.png"), a["crop"])
    cv2.imwrite(os.path.join(d, "mask.png"), a["mask"])
    draw_piece_png(shape, dia, a["color"], os.path.join(d, "drawing.png"))
    import datetime as _dt
    rows, cols = dims_of(shape)
    details = describe_diamonds(a)
    meta = {
        "id": rid,
        "kind": kind,
        "shape": [list(c) for c in shape],
        "rows": rows,
        "cols": cols,
        "block_count": len(shape),
        "diamond_cells": [list(c) for c in dia],
        "diamond_details": [
            {"cell": d["cell"], "class": d["class"],
             "appearance": d["appearance"]} for d in details
        ],
        "source_slot": slot,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pitch": round(float(a["pitch"]), 2),
        "dx": round(float(a["dx"]), 2),
        "dy": round(float(a["dy"]), 2),
        "confidence": round(float(a["score"]), 4),
        "frame_size": list(a["frame_size"]),
        "scale": round(float(a["scale"]), 4),
        "dominant_color_bgr": list(map(int, a["color"])),
    }
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    idx["records"].append({"id": rid, "kind": kind, "shape": meta["shape"],
                           "diamond_cells": meta["diamond_cells"],
                           "block_count": meta["block_count"]})
    _save_index(idx)
    try:
        from .visual_memory import record_diamond_type
        cells_px = a.get("cells_px") or []
        order = a.get("cell_order") or []
        half = int(a.get("diamond_half_px") or 8)
        pos = {tuple(order[i]): cells_px[i]
               for i in range(min(len(order), len(cells_px)))}
        crop = a.get("crop")
        for det in details:
            key = tuple(det["cell"])
            cell_crop = None
            if crop is not None and key in pos:
                cx, cy = int(pos[key][0]), int(pos[key][1])
                hs = max(2, half)
                Hc, Wc = crop.shape[:2]
                x1, x2 = max(0, cx - hs), min(Wc, cx + hs + 1)
                y1, y2 = max(0, cy - hs), min(Hc, cy + hs + 1)
                cell_crop = crop[y1:y2, x1:x2]
            record_diamond_type(det["class"] or "desconhecido", cell_crop,
                                det["appearance"])
    except Exception:
        pass
    return rid, d
