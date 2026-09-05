#!/usr/bin/env python3
"""
Memoria visual OBSERVACIONAL do bot (Blocks of Bitcoin).

Registra variacoes visuais encontradas em testes/execucao SEM alterar
o comportamento do bot: nao modifica catalogo, detector, solver ou controle.

Fluxo exigido: OBSERVAR -> REGISTRAR -> AGRUPAR -> VALIDAR.
Somente exemplos confirmados poderao, no futuro, virar templates/aparencias.

Categorias (nunca confundir):
  A) NOVA FORMA GEOMETRICA  -> disposicao de celulas fora do catalogo
  B) NOVA APARENCIA         -> mesma forma, outra cor/tom/diamante/brilho
  C) VARIACAO DE CAPTURA    -> escala/deslocamento/ruido/resolucao

Estrutura em disco:
  data/visual_memory/
    pieces/      exemplar por grupo (crop/mask/signal)
    diamonds/    recortes de blocos com diamante
    board/       recortes de tabuleiro
    metadata.jsonl   uma linha JSON por ocorrencia (com group_id e contador)

Agrupamento: mesma chave (kind + forma normalizada + bucket de aparencia +
padrao de diamante) reutiliza o grupo e incrementa ocorrencias, em vez de
criar duplicatas.
"""

import os
import json
import time
import hashlib
import datetime

import cv2
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "visual_memory")
PIECES_DIR = os.path.join(BASE_DIR, "pieces")
DIAMONDS_DIR = os.path.join(BASE_DIR, "diamonds")
BOARD_DIR = os.path.join(BASE_DIR, "board")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.jsonl")
DIAMOND_TYPES_DIR = os.path.join(BASE_DIR, "diamonds", "types")
DIAMOND_TYPES_INDEX = os.path.join(DIAMOND_TYPES_DIR, "types.json")


def _ensure_dirs():
    for d in (PIECES_DIR, DIAMONDS_DIR, BOARD_DIR):
        os.makedirs(d, exist_ok=True)


def _load_groups():
    """Indice group_id -> {count, exemplar} a partir do metadata existente."""
    groups = {}
    if not os.path.exists(METADATA_PATH):
        return groups
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                gid = rec.get("group_id")
                if not gid:
                    continue
                g = groups.setdefault(gid, {"count": 0, "exemplar": rec.get("exemplar")})
                g["count"] += 1
                if rec.get("exemplar"):
                    g["exemplar"] = rec["exemplar"]
    except OSError:
        pass
    return groups


def appearance_bucket(crop_bgr, mask):
    """Bucket de aparencia (heuristica v0): HSV medio quantizado dos pixels de bloco."""
    if crop_bgr is None or mask is None:
        return {"h": -1, "s": -1, "v": -1}
    m = (mask > 0)
    if not bool(m.any()):
        return {"h": -1, "s": -1, "v": -1}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = float(hsv[:, :, 0][m].mean())
    s = float(hsv[:, :, 1][m].mean())
    v = float(hsv[:, :, 2][m].mean())
    return {"h": int(h // 15), "s": int(s // 32), "v": int(v // 32)}


def detect_diamond_cells(crop_bgr, cells_px, half_px):
    """Diamante por celula, MESMA especificacao calibrada de vision.py.

    Usa DIAMOND_H_LO/HI, DIAMOND_S_MIN/V_MIN, RESCUE_PINK_MIN/GRAY_MIN e
    RESCUE_REGION_FRAC importados de vision (fonte unica; se a calibracao
    mudar la, o catalogo acompanha). Avalia centro + 4 sub-posicoes
    (+-pitch/4) com a conjuncao rosa-sobre-cinza em cada uma; dispara se
    QUALQUER sub-posicao passar (o icone pode estar ate meia celula
    deslocado do centro amostrado).
    cells_px: lista de (cx, cy) em coordenadas do crop.
    Retorna {idx: {'has_diamond': bool, 'ratio': float, 'class': str|None}}.
    """
    from .vision import (DIAMOND_H_LO, DIAMOND_H_HI, DIAMOND_S_MIN,
                        DIAMOND_V_MIN, RESCUE_PINK_MIN, RESCUE_GRAY_MIN,
                        RESCUE_REGION_FRAC, DIAMOND_CLASSES,
                        DIAMOND_MINIMUMS)
    out = {}
    if crop_bgr is None or not cells_px:
        return out
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float),
               hsv[:, :, 2].astype(float))
    pink_m = ((h >= DIAMOND_H_LO) & (h <= DIAMOND_H_HI)
              & (s > DIAMOND_S_MIN) & (v > DIAMOND_V_MIN))
    for cls in DIAMOND_CLASSES:
        if cls["name"] == "emerald":
            continue  # janela legada ja aplicada acima, bit-identica
        pink_m = (pink_m | ((h >= cls["h_lo"]) & (h <= cls["h_hi"])
                            & (s > cls["s_min"]) & (v > cls["v_min"])))
    cls_masks = [((h >= cls["h_lo"]) & (h <= cls["h_hi"])
                  & (s > cls["s_min"]) & (v > cls["v_min"]), cls["name"])
                 for cls in DIAMOND_CLASSES]
    gray_m = ((s < 65) & (v >= 110))
    H, W = h.shape[:2]
    hs = max(2, int(half_px * RESCUE_REGION_FRAC))
    q = max(7, int(round(half_px / 2.0)))
    for i, (cx, cy) in enumerate(cells_px):
        best_pk = 0.0
        fired = False
        fired_cls = None
        fire_frac = 0.0
        center_fire = False
        center_frac = 0.0
        for ox, oy in ((0, 0), (q, 0), (-q, 0), (0, q), (0, -q)):
            x1, x2 = max(0, cx + ox - hs), min(W, cx + ox + hs + 1)
            y1, y2 = max(0, cy + oy - hs), min(H, cy + oy + hs + 1)
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            pk = float(pink_m[y1:y2, x1:x2].mean())
            gr = float(gray_m[y1:y2, x1:x2].mean())
            best_pk = max(best_pk, pk)
            for (m, name) in cls_masks:
                f = float(m[y1:y2, x1:x2].mean())
                pmin, gmin = DIAMOND_MINIMUMS[name]
                if f >= pmin and gr >= gmin:
                    if not fired:
                        fired = True
                        fired_cls = name
                        fire_frac = f
                    if ox == 0 and oy == 0:
                        center_fire = True
                        center_frac = max(center_frac, f)
        out[i] = {"has_diamond": bool(fired), "ratio": round(best_pk, 4),
                  "class": fired_cls, "_fire": round(fire_frac, 4),
                  "_center": center_fire,
                  "_center_frac": round(center_frac, 4)}
    # Supressao de respingo: flag só-de-offset cai se uma célula vizinha
    # (centros a ~1 pitch; limiar 1.3 exclui diagonais) dispara NO CENTRO
    # com fracao dominante (>=1.25x). Medido: respingo 0.058 vs centro
    # 0.086 (cai); esmeraldas 0.195 vs 0.195 ( ficam); sem centro aceso
    # (caso 018) nada muda. Sem centro aceso em X, sem queda.
    import math as _math
    pitch_est = 2 * half_px
    for i, (cx, cy) in enumerate(cells_px):
        if not out[i]["has_diamond"] or out[i]["_center"]:
            continue
        for j, (dx_, dy_) in enumerate(cells_px):
            if i == j or not out[j]["has_diamond"] or not out[j]["_center"]:
                continue
            if _math.hypot(cx - dx_, cy - dy_) > 1.3 * pitch_est:
                continue
            if out[j]["_center_frac"] >= 1.25 * out[i]["_fire"]:
                out[i] = {"has_diamond": False, "ratio": out[i]["ratio"],
                          "class": None, "_fire": out[i]["_fire"],
                          "_center": False,
                          "_center_frac": out[i]["_center_frac"]}
                break
    for v in out.values():
        v.pop("_fire", None)
        v.pop("_center", None)
        v.pop("_center_frac", None)
    return out


def diamond_cell_appearance(crop_bgr, cx, cy, half_px, cls_name):
    """Aparência visual do diamante numa célula (somente leitura).

    Mede os pixels da máscara da CLASSE na janela da célula: BGR/HSV
    dominantes (mediana), brilho médio (V) e contagem de pixels.
    Retorna dict ou None se nada da classe na janela.
    """
    from .vision import DIAMOND_CLASSES
    if crop_bgr is None:
        return None
    spec = next((c for c in DIAMOND_CLASSES if c["name"] == cls_name), None)
    if spec is None:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float),
               hsv[:, :, 2].astype(float))
    m = ((h >= spec["h_lo"]) & (h <= spec["h_hi"])
         & (s > spec["s_min"]) & (v > spec["v_min"]))
    H, W = h.shape[:2]
    hs = max(2, int(half_px))
    x1, x2 = max(0, cx - hs), min(W, cx + hs + 1)
    y1, y2 = max(0, cy - hs), min(H, cy + hs + 1)
    sel = m[y1:y2, x1:x2]
    if not bool(sel.any()):
        return None
    px_bgr = crop_bgr[y1:y2, x1:x2][sel]
    px_hsv = np.stack([h[y1:y2, x1:x2][sel], s[y1:y2, x1:x2][sel],
                       v[y1:y2, x1:x2][sel]], axis=1)
    return {"class": cls_name,
            "dominant_bgr": [int(x) for x in np.median(px_bgr, axis=0)],
            "dominant_hsv": [int(x) for x in np.median(px_hsv, axis=0)],
            "brightness_v": round(float(px_hsv[:, 2].mean()), 1),
            "pixel_count": int(sel.sum())}


def diamond_type_index():
    """Índice somente-leitura dos tipos visuais de diamante registrados."""
    if not os.path.exists(DIAMOND_TYPES_INDEX):
        return {"types": {}}
    try:
        with open(DIAMOND_TYPES_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"types": {}}


def _save_diamond_types(idx):
    os.makedirs(DIAMOND_TYPES_DIR, exist_ok=True)
    with open(DIAMOND_TYPES_INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def record_diamond_type(name, cell_crop_bgr, appearance):
    """Registra observação de um tipo visual (somente data/).

    Primeira vez guarda exemplar PNG + aparência; depois só incrementa
    ocorrências. Retorna a entrada do tipo. Nunca levanta exceção.
    """
    try:
        idx = diamond_type_index()
        entry = idx["types"].get(name)
        if entry is None:
            entry = {"name": name, "count": 0, "exemplar": None,
                     "appearance": appearance,
                     "first_seen": datetime.datetime.now().isoformat(
                         timespec="seconds")}
            idx["types"][name] = entry
        entry["count"] = int(entry.get("count", 0)) + 1
        if not entry.get("exemplar") and cell_crop_bgr is not None:
            os.makedirs(DIAMOND_TYPES_DIR, exist_ok=True)
            safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                           for c in name)[:32] or "tipo"
            path = os.path.join(DIAMOND_TYPES_DIR, f"{safe}.png")
            cv2.imwrite(path, cell_crop_bgr)
            entry["exemplar"] = os.path.basename(path)
        _save_diamond_types(idx)
        return entry
    except Exception:
        return None


def _group_key(kind, shape_norm, appearance, diamond_pattern):
    raw = json.dumps({
        "kind": kind,
        "shape": sorted([list(c) for c in shape_norm]),
        "appearance": appearance,
        "diamonds": diamond_pattern,
    }, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def record_observation(kind, slot, crop_bgr, mask, signal_region,
                       pitch, dx, dy, shape_cells, n_blocks,
                       diamond_cells, confidence, final_result,
                       extra=None):
    """Registra uma observacao (somente escrita em disco; sem efeito no bot).

    Retorna o registro criado (dict) ou None em caso de falha de IO.
    Nunca levanta excecao para nao quebrar o fluxo de diagnostico.
    """
    try:
        _ensure_dirs()
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        shape_norm = sorted([tuple(c) for c in (shape_cells or [])])
        appearance = appearance_bucket(crop_bgr, mask)
        diamond_pattern = sorted([i for i, d in (diamond_cells or {}).items()
                                  if d.get("has_diamond")])
        gid = _group_key(kind, shape_norm, appearance, diamond_pattern)
        groups = _load_groups()
        is_new = gid not in groups
        count = groups.get(gid, {"count": 0})["count"] + 1

        subdir = {"piece": PIECES_DIR, "diamond": DIAMONDS_DIR,
                  "board": BOARD_DIR}.get(kind, PIECES_DIR)
        exemplar = groups.get(gid, {}).get("exemplar")
        if is_new:
            exemplar = {
                "crop": f"{gid}_crop.png",
                "mask": f"{gid}_mask.png",
                "signal": f"{gid}_signal.npy",
            }
            if crop_bgr is not None:
                cv2.imwrite(os.path.join(subdir, exemplar["crop"]), crop_bgr)
            if mask is not None:
                cv2.imwrite(os.path.join(subdir, exemplar["mask"]), mask)
            if signal_region is not None:
                np.save(os.path.join(subdir, exemplar["signal"]),
                        np.asarray(signal_region, dtype=np.float32))

        rec = {
            "ts": ts,
            "kind": kind,            # piece | diamond | board
            "category_hint": None,   # preenchido na curadoria: A | B | C
            "slot": slot,
            "group_id": gid,
            "group_count": count,
            "new_group": bool(is_new),
            "exemplar": exemplar,
            "pitch": None if pitch is None else round(float(pitch), 2),
            "dx": None if dx is None else round(float(dx), 2),
            "dy": None if dy is None else round(float(dy), 2),
            "shape": [list(c) for c in shape_norm],
            "n_blocks": int(n_blocks) if n_blocks is not None else None,
            "diamonds": {str(k): v for k, v in (diamond_cells or {}).items()},
            "appearance": appearance,
            "confidence": None if confidence is None else round(float(confidence), 4),
            "final_result": final_result,
            "extra": extra or {},
        }
        with open(METADATA_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    except Exception:
        return None
