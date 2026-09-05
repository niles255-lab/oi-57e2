#!/usr/bin/env python3
"""
Rastreamento isolado de peca com diamante.
Etapas: crop -> mask -> signal -> celulas -> pre-snap -> pos-snap.
NAO altera vision.py/bot.py. Nenhuma jogada real.
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import adb_io
from vision import (
    _build_signal_map, _block_mask, _snap_to_catalog, _normalize,
    _max_run, read_pieces, NATIVE_DEVICE_W, NATIVE_DEVICE_H,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_diamond")
os.makedirs(OUT, exist_ok=True)

# Experimento controlado de offsets (diagnóstico apenas; vision.py intacto).
# Limites testados em sequência; passo 3px mantido como no detector.
DXDY_LIMITS = [(-12.0, 12.0), (-8.0, 8.0)]


def main():
    print("== rastreamento do diamante ==", flush=True)
    ok = adb_io.init_scrcpy_stream()
    print(f"stream: {ok}", flush=True)
    if not ok:
        time.sleep(2)
        adb_io.stop_scrcpy_stream()
        ok = adb_io.init_scrcpy_stream()
        print(f"stream (retry): {ok}", flush=True)
    time.sleep(2)
    fr = adb_io.capture_frame()
    if fr is None:
        print("ERRO: nenhum frame capturado (stream e screencap falharam)", flush=True)
        adb_io.stop_scrcpy_stream()
        return
    print(f"frame: {tuple(fr.shape)}", flush=True)
    cv2.imwrite(os.path.join(OUT, "frame.png"), fr)

    sig = _build_signal_map(fr)
    H, W = sig.shape[:2]
    sx = W / NATIVE_DEVICE_W
    print(f"escala: {sx:.4f}", flush=True)

    # encontra slot com diamante: procura regiao da bandeja com S baixo + V alto + rosa
    pieces = read_pieces(fr)
    for i, p in enumerate(pieces):
        if p and p.cells:
            print(f"slot {i}: {p.width}x{p.height} blocos={len(p.cells)} cells={sorted(p.cells)}", flush=True)
        else:
            print(f"slot {i}: None", flush=True)

    # experimento controlado de offsets (passo 3px mantido)
    for lo, hi in DXDY_LIMITS:
        print(f"\n{'#'*70}\nEXPERIMENTO dx,dy em [{lo:+.0f},{hi:+.0f}] (passo 3px)\n{'#'*70}", flush=True)
        for slot in range(3):
            trace_slot(fr, sig, slot, dxdy_lim=(lo, hi))

    adb_io.stop_scrcpy_stream()
    print(f"imagens em: {OUT}", flush=True)


def eval_fixed_geometry(sig, W, H, sx, sy, sc, pitch_native):
    """Candidato ancorado na geometria calibrada: pitch=58*sc, dx=0, dy=0.

    Responde: o candidato correto sequer e gerado? Se gerado, quanto pontua?
    """
    pitch = 58.0 * sc
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    xi = np.round(sx + cc[None, :] * pitch).astype(np.int32)
    yi = np.round(sy + rr[:, None] * pitch).astype(np.int32)
    xo = (xi < 0) | (xi >= W)
    yo = (yi < 0) | (yi >= H)
    xic = np.clip(xi, 0, W - 1)
    yic = np.clip(yi, 0, H - 1)
    vals = sig[yic[:, :, np.newaxis], xic[np.newaxis, :, :]].astype(np.float64)
    oob = xo[np.newaxis, :, :] | yo[:, :, np.newaxis]
    vals = np.where(oob, 0.0, vals)
    strict = vals >= 0.60
    adj = np.zeros_like(strict)
    adj[:-1, :, :] |= strict[1:, :, :]
    adj[1:, :, :] |= strict[:-1, :, :]
    adj[:, :, :-1] |= strict[:, :, 1:]
    adj[:, :, 1:] |= strict[:, :, :-1]
    bits = (strict | ((vals >= 0.40) & adj))[:, 0, :]
    vals = vals[:, 0, :]
    rw = np.where(bits.any(axis=1))[0]
    cw_ = np.where(bits.any(axis=0))[0]
    if rw.size == 0 or cw_.size == 0:
        print("  geometria calibrada (58,dx=0,dy=0): NENHUM bit -> candidato nem gerado", flush=True)
        return None
    r0, r1 = int(rw[0]), int(rw[-1]) + 1
    c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
    bb = bits[r0:r1, c0:c1]
    bv = vals[r0:r1, c0:c1]
    n = int(bb.sum())
    det = [(r, c) for r in range(r1 - r0) for c in range(c1 - c0) if bb[r, c]]
    if n < 2 or n > 5:
        print(f"  geometria calibrada: n={n} fora de [2,5] cells={sorted(_normalize(det)) if det else det}", flush=True)
        return None
    ms = float(bv[bb].mean())
    bp = np.pad(bb.astype(np.int8), 1)
    nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
    support = int(((nbrs > 0) & bb).sum())
    run = _max_run(bb)
    eib = (r1 - r0) * (c1 - c0) - n
    sn = _snap_to_catalog(det)
    jac = 0.0
    if len(sn) == n:
        rn_, sn_ = _normalize(det), _normalize(sn)
        jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
    score = (ms + 0.050 * support + 0.040 * n + 0.100 * run
             - 0.080 * eib - 1.0 * 0.0 - 0.15 * (5 - n) + 0.20 * jac)
    print(f"  geometria calibrada (58,dx=0,dy=0): n={n} cells={sorted(_normalize(det))} "
          f"mean={ms:.3f} support={support} run={run} empty={eib} jaccard={jac:.3f} "
          f"SCORE={score:.4f} (pitch_err=0)", flush=True)
    return (n, sorted(_normalize(det)), score)


def trace_slot(frame, sig, slot, dxdy_lim=None):
    print(f"\n{'='*60}\nSLOT {slot}\n{'='*60}", flush=True)
    H, W = sig.shape[:2]
    sxn, syn = config.TRAY_SLOT_CENTERS[slot]
    sc = W / NATIVE_DEVICE_W
    sx, sy = int(sxn * sc), int(syn * (H / NATIVE_DEVICE_H))
    half = int(config.PIECE_SLOT_HALF * sc)
    x1, x2 = max(0, sx - half), min(W, sx + half)
    y1, y2 = max(0, sy - half), min(H, sy + half)

    lim_tag = f"lim{int(abs(dxdy_lim[0]))}" if dxdy_lim is not None else "ful"
    crop = frame[y1:y2, x1:x2].copy()
    print(f"crop regiao: x=[{x1}:{x2}] y=[{y1}:{y2}] shape={tuple(crop.shape)}", flush=True)
    cv2.imwrite(os.path.join(OUT, f"slot{slot}_1_crop.png"), crop)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _block_mask(hsv)
    cv2.imwrite(os.path.join(OUT, f"slot{slot}_2_mask.png"), mask)
    print(f"mask branca: {(mask > 0).mean():.3f}", flush=True)

    # signal map na regiao
    sreg = sig[y1:y2, x1:x2]
    print(f"signal regiao: min={sreg.min():.4f} max={sreg.max():.4f} mean={sreg.mean():.4f}", flush=True)

    # HSV no centro do crop (possivel diamante) vs bordas
    hh, ww = crop.shape[:2]
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    print(f"S: min={int(s_ch.min())} max={int(s_ch.max())} mean={s_ch.mean():.1f} | "
          f"V: min={int(v_ch.min())} max={int(v_ch.max())} mean={v_ch.mean():.1f}", flush=True)

    # grade estimada de celulas: replica EXATA da busca do detect_piece,
    # incluindo score total, gate de snap e Jaccard.
    # Faixa e peso DEVEM acompanhar vision.py (52-64, peso pitch 1.0).
    pitch_native = config.TRAY_CELL_PITCH
    PITCH_MIN, PITCH_MAX, PITCH_STEP = 52.0 * sc, 64.0 * sc, 1.0 * sc
    PITCH_W = 1.0
    best = None
    best_score = -1.0
    n_avaliados = 0
    n_rej_sig = n_rej_snap_n = n_rej_jac = 0
    lo, hi = dxdy_lim if dxdy_lim is not None else (-1e9, 1e9)
    for pitch in np.arange(PITCH_MIN, PITCH_MAX, PITCH_STEP):
        half_p = int(pitch / 2) + 2
        dx = np.arange(max(-half_p, lo), min(half_p, hi) + 1, 3, dtype=np.float64)
        dy = np.arange(max(-half_p, lo), min(half_p, hi) + 1, 3, dtype=np.float64)
        if dx.size == 0 or dy.size == 0:
            continue
        cc = np.arange(7, dtype=np.float64) - 3.0
        rr = np.arange(7, dtype=np.float64) - 3.0
        xi = np.round(sx + dx[:, None] + cc[None, :] * pitch).astype(np.int32)
        yi = np.round(sy + dy[:, None] + rr[None, :] * pitch).astype(np.int32)
        xo = (xi < 0) | (xi >= W)
        yo = (yi < 0) | (yi >= H)
        xic = np.clip(xi, 0, W - 1)
        yic = np.clip(yi, 0, H - 1)
        vals = sig[yic[np.newaxis, :, :, np.newaxis], xic[:, np.newaxis, np.newaxis, :]].astype(np.float64)
        vals[(xo[:, None, None, :] | yo[None, :, :, None])] = 0.0
        strict = vals >= 0.60
        adj = np.zeros_like(strict)
        adj[:, :, :-1, :] |= strict[:, :, 1:, :]
        adj[:, :, 1:, :] |= strict[:, :, :-1, :]
        adj[:, :, :, :-1] |= strict[:, :, :, 1:]
        adj[:, :, :, 1:] |= strict[:, :, :, :-1]
        bits = strict | ((vals >= 0.40) & adj)
        for idx in np.argwhere(bits.any(axis=(-1, -2))):
            i_dx, j_dy = int(idx[0]), int(idx[1])
            b = bits[i_dx, j_dy]
            v = vals[i_dx, j_dy]
            rw = np.where(b.any(axis=1))[0]
            cw_ = np.where(b.any(axis=0))[0]
            if rw.size == 0 or cw_.size == 0:
                continue
            r0, r1 = int(rw[0]), int(rw[-1]) + 1
            c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
            bb = b[r0:r1, c0:c1]
            bv = v[r0:r1, c0:c1]
            n = int(bb.sum())
            if n < 2 or n > 5:
                continue
            n_avaliados += 1
            ms = float(bv[bb].mean())
            if ms < 0.55:
                n_rej_sig += 1
                continue
            bp = np.pad(bb.astype(np.int8), 1)
            nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            support = int(((nbrs > 0) & bb).sum())
            run = _max_run(bb)
            bh2, bw2 = r1 - r0, c1 - c0
            empty_in_bbox = bh2 * bw2 - n
            det = [(r, c) for r in range(bh2) for c in range(bw2) if bb[r, c]]
            snapped = _snap_to_catalog(det)
            if len(snapped) != n:
                n_rej_snap_n += 1
                continue
            rn_ = _normalize(det)
            sn_ = _normalize(snapped)
            jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
            if jac < 0.70:
                n_rej_jac += 1
                continue
            pitch_err = abs(pitch - pitch_native * sc) / pitch_native
            score = (ms + 0.050 * support + 0.040 * n + 0.100 * run
                     - 0.080 * empty_in_bbox - PITCH_W * pitch_err
                     - 0.15 * (5 - n) + 0.20 * jac)
            if score > best_score:
                best_score = score
                pts = np.zeros((r1 - r0, c1 - c0, 2), dtype=int)
                for rr_i in range(r0, r1):
                    for cc_i in range(c0, c1):
                        pts[rr_i - r0, cc_i - c0] = (int(xic[i_dx, cc_i]), int(yic[j_dy, rr_i]))
                best = (ms, pitch, float(dx[i_dx]), float(dy[j_dy]),
                        bb.copy(), bv.copy(), pts, (r0, r1, c0, c1),
                        support, run, empty_in_bbox, jac, score)
    print(f"candidatos 2-5 blocos: {n_avaliados} | rej mean_sig: {n_rej_sig} | "
          f"rej snap_n: {n_rej_snap_n} | rej jaccard: {n_rej_jac}", flush=True)
    if best is None:
        print("nenhum candidato passou nos gates (score final <0.75 ou vazio)", flush=True)
        return
    ms, pitch, dxb, dyb, bb, bv, pts, (r0, r1, c0, c1), support, run, eib, jac, score = best
    print(f"VENCEDOR pitch={pitch:.2f} dx={dxb:.1f} dy={dyb:.1f} bbox={bb.shape} "
          f"grade=[{r0}:{r1},{c0}:{c1}] mean_sig={ms:.4f} support={support} run={run} "
          f"empty={eib} jaccard={jac:.3f} SCORE={score:.4f}", flush=True)

    # tabela por celula do bbox: bit | valor signal | pixel amostrado
    bh, bw = bb.shape
    print("tabela por celula -> bit : signal : (x,y) no frame:", flush=True)
    for r in range(bh):
        row = []
        for c in range(bw):
            row.append(f"{int(bb[r, c])}:{float(bv[r, c]):.2f}:({pts[r, c, 0]},{pts[r, c, 1]})")
        print(f"  linha {r}: " + " ".join(row), flush=True)

    print("--- candidato ancorado na calibracao ---", flush=True)
    eval_fixed_geometry(sig, W, H, sx, sy, sc, config.TRAY_CELL_PITCH)

    cells_pre = [(r, c) for r in range(bh) for c in range(bw) if bb[r, c]]
    print(f"celulas PRE-snap ({len(cells_pre)}): {sorted(_normalize(cells_pre))}", flush=True)
    snapped = _snap_to_catalog(list(cells_pre))
    print(f"celulas POS-snap ({len(snapped)}): {sorted(_normalize(snapped)) if snapped else snapped}", flush=True)
    if sorted(_normalize(cells_pre)) != sorted(_normalize(snapped)):
        print(">>> SNAP ALTEROU A FORMA", flush=True)
    else:
        print("snap manteve a forma", flush=True)

    # memoria visual observacional (nao altera o bot; so registra em disco)
    try:
        from visual_memory import record_observation, detect_diamond_cells
        occ_idx, cells_px = [], []
        for r in range(bh):
            for c in range(bw):
                if bb[r, c]:
                    occ_idx.append((r, c))
                    cells_px.append((int(pts[r, c, 0]) - x1, int(pts[r, c, 1]) - y1))
        dia = detect_diamond_cells(crop, cells_px, pitch / 2.0)
        dia = {occ_idx[k]: v for k, v in dia.items()}
        rec = record_observation(
            kind="piece", slot=slot, crop_bgr=crop, mask=mask,
            signal_region=sig[y1:y2, x1:x2], pitch=pitch, dx=dxb, dy=dyb,
            shape_cells=cells_pre, n_blocks=len(cells_pre),
            diamond_cells=dia, confidence=score,
            final_result={"cells": [list(c) for c in snapped], "n_blocks": len(snapped)},
            extra={"dxdy_lim": list(dxdy_lim) if dxdy_lim else None,
                   "grade": [r0, r1, c0, c1], "mean_sig": round(ms, 4),
                   "support": support, "run": run, "empty": eib,
                   "jaccard": round(jac, 4)},
        )
        if rec is not None:
            print(f"[memoria] grupo={rec['group_id']} novo={rec['new_group']} "
                  f"ocorrencias={rec['group_count']} diamantes={sorted([k for k, v in dia.items() if v.get('has_diamond')])}",
                  flush=True)
    except Exception as e:
        print(f"[memoria] falha ao registrar (sem efeito no teste): {e}", flush=True)

    # comparacao: melhor candidato com o pitch ANTIGO vencedor (50px),
    # avaliado com os pesos NOVOS — prova de que a geometria agora decide
    p50 = 50.0 * sc
    best50, best50_score = None, float("-inf")
    half_p = int(p50 / 2) + 2
    dx = np.arange(-half_p, half_p + 1, 3, dtype=np.float64)
    dy = np.arange(-half_p, half_p + 1, 3, dtype=np.float64)
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    xi = np.round(sx + dx[:, None] + cc[None, :] * p50).astype(np.int32)
    yi = np.round(sy + dy[:, None] + rr[None, :] * p50).astype(np.int32)
    xo = (xi < 0) | (xi >= W)
    yo = (yi < 0) | (yi >= H)
    xic = np.clip(xi, 0, W - 1)
    yic = np.clip(yi, 0, H - 1)
    vals = sig[yic[np.newaxis, :, :, np.newaxis], xic[:, np.newaxis, np.newaxis, :]].astype(np.float64)
    vals[(xo[:, None, None, :] | yo[None, :, :, None])] = 0.0
    strict = vals >= 0.60
    adj = np.zeros_like(strict)
    adj[:, :, :-1, :] |= strict[:, :, 1:, :]
    adj[:, :, 1:, :] |= strict[:, :, :-1, :]
    adj[:, :, :, :-1] |= strict[:, :, :, 1:]
    adj[:, :, :, 1:] |= strict[:, :, :, :-1]
    bits = strict | ((vals >= 0.40) & adj)
    for idx in np.argwhere(bits.any(axis=(-1, -2))):
        i_dx, j_dy = int(idx[0]), int(idx[1])
        b = bits[i_dx, j_dy]
        v = vals[i_dx, j_dy]
        rw = np.where(b.any(axis=1))[0]
        cw_ = np.where(b.any(axis=0))[0]
        if rw.size == 0 or cw_.size == 0:
            continue
        r0, r1 = int(rw[0]), int(rw[-1]) + 1
        c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
        bb50 = b[r0:r1, c0:c1]
        bv50 = v[r0:r1, c0:c1]
        n50 = int(bb50.sum())
        if n50 < 2 or n50 > 5:
            continue
        ms50 = float(bv50[bb50].mean())
        if ms50 < 0.55:
            continue
        bp50 = np.pad(bb50.astype(np.int8), 1)
        nb50 = (bp50[:-2, 1:-1] + bp50[2:, 1:-1] + bp50[1:-1, :-2] + bp50[1:-1, 2:])
        sup50 = int(((nb50 > 0) & bb50).sum())
        run50 = _max_run(bb50)
        bh50, bw50 = r1 - r0, c1 - c0
        eib50 = bh50 * bw50 - n50
        det50 = [(r, c) for r in range(bh50) for c in range(bw50) if bb50[r, c]]
        sn50 = _snap_to_catalog(det50)
        if len(sn50) != n50:
            continue
        rn50 = _normalize(det50)
        snn50 = _normalize(sn50)
        jac50 = len(rn50 & snn50) / len(rn50 | snn50) if (rn50 | snn50) else 0
        if jac50 < 0.70:
            continue
        err50 = abs(p50 - pitch_native * sc) / pitch_native
        s50 = (ms50 + 0.050 * sup50 + 0.040 * n50 + 0.100 * run50
               - 0.080 * eib50 - PITCH_W * err50 - 0.15 * (5 - n50) + 0.20 * jac50)
        if s50 > best50_score:
            best50_score = s50
            best50 = (n50, sorted(_normalize(det50)), ms50, sup50, run50, eib50, jac50, err50)
    if best50 is None:
        print("pitch=50: nenhum candidato valido", flush=True)
    else:
        n50, cells50, ms50, sup50, run50, eib50, jac50, err50 = best50
        print(f"pitch=50 (antigo vencedor): n={n50} cells={cells50} mean={ms50:.3f} "
              f"pitch_err={err50:.3f} penalidade={PITCH_W * err50:.3f} SCORE={best50_score:.4f} "
              f"-> {'PERDE' if best50_score < score else 'VENCERIA (PROBLEMA!)'}", flush=True)

    # visual: marca os pontos amostrados sobre o crop
    vis = crop.copy()
    for r in range(bh):
        for c in range(bw):
            px, py = int(pts[r, c, 0]) - x1, int(pts[r, c, 1]) - y1
            if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                color = (0, 255, 0) if bb[r, c] else (0, 0, 255)
                cv2.circle(vis, (px, py), 6, color, 2)
                cv2.putText(vis, f"{bv[r, c]:.2f}", (px + 8, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"slot{slot}_3_pontos_{lim_tag}.png"), vis)
    print(f"bbox {bh}x{bw} sobre crop {crop.shape[1]}x{crop.shape[0]} "
          f"-> slot{slot}_3_pontos_{lim_tag}.png (verde=1, vermelho=0)", flush=True)


if __name__ == "__main__":
    main()
