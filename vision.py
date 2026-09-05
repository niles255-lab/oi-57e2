"""
Deteccao de tabuleiro e pecas via analise de imagem.

Abordagem geometrica (sem templates):
  - Mascara HSV cobre blocos coloridos E blocos cinza/prata (diamantes)
  - Mapa de sinal pre-computado por boxFilter para velocidade O(1) por celula
  - Scan sobre pitch e offset ao redor de cada slot
  - Score favorece runs longos (corrige barra de 5) e amostragem de area
    em vez de ponto central (corrige icone de diamante no centro do bloco)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

import config
from model import Board, Piece, PIECE_CATALOG

# Resolução nativa do dispositivo (Redmi Note 8)
NATIVE_DEVICE_W = 1080
NATIVE_DEVICE_H = 2340

# ---------------------------------------------------------------------------
# Assinatura do diamante para resgate LOCALIZADO (sem adjacencia global).
# Calibrado no frame live_20260904_185226: bloco com diamante tem
# pink>=0.27 na regiao central, enquanto triangulo/roxo/vazio tem 0.000.
# O resgate so avalia celulas 0-bit vizinhas a celulas strict 1-bit e
# exige tambem face de bloco (gray) — as duas condicoes juntas.
# ---------------------------------------------------------------------------
DIAMOND_H_LO, DIAMOND_H_HI = 152, 172
DIAMOND_S_MIN, DIAMOND_V_MIN = 80, 100
RESCUE_PINK_MIN = 0.05
RESCUE_GRAY_MIN = 0.20
RESCUE_REGION_FRAC = 0.6  # lado do quadrado central como fracao do pitch

# Classes de diamante (o jogo varia forma e cor; cada classe tem sua
# janela de cor, mas TODAS exigem a mesma conjuncao rosa-sobre-cinza).
# 'emerald' = janela legada (comportamento original preservado);
# 'purple' = triangulo roxo, nucleo medido em 2026-09-05 (H 134-145,
# S~184, V~186). Tabela extensivel: nova cor = nova linha.
DIAMOND_CLASSES = (
    {"name": "emerald", "h_lo": 152, "h_hi": 172, "s_min": 80, "v_min": 100,
     "pink_min": 0.05, "gray_min": 0.20},
    {"name": "purple", "h_lo": 130, "h_hi": 150, "s_min": 80, "v_min": 120,
     "pink_min": 0.10, "gray_min": 0.12},
    {"name": "gold", "h_lo": 35, "h_hi": 55, "s_min": 80, "v_min": 120,
     "pink_min": 0.10, "gray_min": 0.20},
)
DIAMOND_MINIMUMS = {c["name"]: (c["pink_min"], c["gray_min"])
                    for c in DIAMOND_CLASSES}


def _diamond_class_masks(hsv):
    """Uma mascara bool por classe de diamante (mesma HSV de entrada)."""
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    out = []
    for cls in DIAMOND_CLASSES:
        m = ((h >= cls["h_lo"]) & (h <= cls["h_hi"])
             & (s > cls["s_min"]) & (v > cls["v_min"]))
        out.append((cls["name"], m))
    return out


def _rescue_maps(frame: np.ndarray):
    """Mapas integrais (rosa-diamante, cinza-bloco) para o resgate localizado.

    Calculado UMA vez por frame em read_pieces(); consultas O(1) por celula.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink = np.zeros(hsv.shape[:2], dtype=bool)
    cls_ii = []
    for name, m in _diamond_class_masks(hsv):
        pink |= m
        cls_ii.append((name, cv2.integral(m.astype(np.float32))))
    pink = pink.astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    gray = (((s < 65) & (v >= 110)).astype(np.float32))
    return cv2.integral(pink), cv2.integral(gray), cls_ii


def _cell_has_diamond(pink_ii, gray_ii, W, H, px, py, pitch, cls_ii=None):
    """Assinatura de diamante na regiao da celula (QUALQUER classe).

    Retorna (ok, pink_frac, gray_frac). Cada classe usa seus próprios
    mínimos (DIAMOND_MINIMUMS), sempre com a conjuncao rosa-sobre-cinza;
    a esmeralda é avaliada primeiro (prioridade legada). Sem cls_ii,
    cai no comportamento legado (janela única + consts). Mesma avaliacao
    de centro + 4 sub-posicoes de sempre.

    Avalia o centro E 4 sub-posicoes (±pitch/4): o icone pode estar ate
    meia celula deslocado do ponto da grade vencedora (erro de fase
    medido: assinatura do diamante entre as linhas y=1698/1756). A celula
    atribuida continua sendo a da grade — so a evidencia e colhida com
    tolerancia. Dispara se QUALQUER sub-posicao passar na conjuncao.
    """
    hs = max(2, int(pitch * RESCUE_REGION_FRAC / 2.0))
    q = max(7, int(round(pitch / 4.0)))
    best = (False, 0.0, 0.0)
    for ox, oy in ((0, 0), (q, 0), (-q, 0), (0, q), (0, -q)):
        x1, x2 = max(0, px + ox - hs), min(W, px + ox + hs + 1)
        y1, y2 = max(0, py + oy - hs), min(H, py + oy + hs + 1)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue
        gr = (gray_ii[y2, x2] - gray_ii[y1, x2]
              - gray_ii[y2, x1] + gray_ii[y1, x1]) / area
        if not cls_ii:
            pk = (pink_ii[y2, x2] - pink_ii[y1, x2]
                  - pink_ii[y2, x1] + pink_ii[y1, x1]) / area
            if pk >= RESCUE_PINK_MIN and gr >= RESCUE_GRAY_MIN:
                return True, float(pk), float(gr)
            if pk > best[1]:
                best = (False, float(pk), float(gr))
            continue
        for name, ii in cls_ii:
            pk = (ii[y2, x2] - ii[y1, x2]
                  - ii[y2, x1] + ii[y1, x1]) / area
            pmin, gmin = DIAMOND_MINIMUMS[name]
            if pk >= pmin and gr >= gmin:
                return True, float(pk), float(gr)
            if pk > best[1]:
                best = (False, float(pk), float(gr))
    return best


def _complete_winner_with_diamonds(sig, W, H, sx, sy, pitch, dx, dy,
                                   r0, c0, best_b, slot_index, maps):
    """Completude localizada sobre o VENCEDOR (mesmo pitch/bbox).

    Testa o anel de celulas 0-bit 4-vizinhas as celulas do vencedor com a
    assinatura de diamante. Nao re-pontua: o vencedor ja foi validado pelo
    score; aqui so se adiciona celula visualmente confirmada como diamante.
    Retorna (new_best_b, new_r0, new_c0, rescued_bbox_rel) ou None quando
    nada a completar ou algum gate (n<=5, snap, Jaccard) falha.
    """
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    xi = np.round(sx + dx + cc * pitch).astype(np.int32)
    yi = np.round(sy + dy + rr * pitch).astype(np.int32)
    occ = {(int(r0 + r), int(c0 + c)) for r, c in zip(*np.where(best_b))}
    fired = []
    for (gr, gc) in sorted(occ):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < 7 and 0 <= nc < 7):
                continue
            if (nr, nc) in occ or (nr, nc) in fired:
                continue
            px, py = int(xi[nc]), int(yi[nr])
            if not (0 <= px < W and 0 <= py < H):
                continue
            ok, pk, grv = _cell_has_diamond(
                maps[0], maps[1], W, H, px, py, pitch,
                maps[2] if len(maps) > 2 else None)
            if ok:
                fired.append((int(nr), int(nc)))
                if config.DEBUG:
                    print(f"  [rescue] slot {slot_index} celula grade "
                          f"({nr},{nc}) diamante pink={pk:.3f} gray={grv:.3f}")
    if not fired:
        return None
    new_occ = set(occ) | set(fired)
    rs = [r for r, _ in new_occ]
    cs = [c for _, c in new_occ]
    nr0, nr1, nc0, nc1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
    nb = np.zeros((nr1 - nr0, nc1 - nc0), dtype=bool)
    for (gr, gc) in new_occ:
        nb[gr - nr0, gc - nc0] = True
    n = len(new_occ)
    if n < 2 or n > (9 if _is_native_scale(W / NATIVE_DEVICE_W, H / NATIVE_DEVICE_H) else 5):
        return None
    det = [(r, c) for r in range(nr1 - nr0) for c in range(nc1 - nc0) if nb[r, c]]
    sn = _snap_to_catalog(det)
    if sn is None or len(sn) != n:
        return None
    rn_ = _normalize(det)
    sn_ = _normalize(sn)
    jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
    if jac < 0.70:
        return None
    rescued_rel = [(r - nr0, c - nc0) for (r, c) in fired]
    return nb, nr0, nc0, rescued_rel


# ---------------------------------------------------------------------------
# Mascara binaria de blocos
# ---------------------------------------------------------------------------

def _block_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Retorna mascara onde blocos reais estao presentes.

    Inclui:
      - Blocos coloridos: alta saturacao (S >= 80) e alto valor (V >= 45)
      - Blocos cinza/prata (contendo icone de diamante): S baixo mas V alto
        Isso corrige o Problema B — um icone de diamante no centro do bloco
        nao deve fazer o bloco desaparecer.

    O fundo da bandeja e o fundo do tabuleiro sao escuros (V < 60),
    portanto o threshold V >= 110 para blocos cinza nao cria falsos positivos.
    """
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    colored    = (s >= 80) & (v >= 45)
    gray_block = (s < 65)  & (v >= 110)
    mask = (colored | gray_block).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


# ---------------------------------------------------------------------------
# Mapa de sinal pre-computado (velocidade: O(N) uma vez por frame)
# ---------------------------------------------------------------------------

# Raio da janela de vizinhanca em pixels (nativo). Calibrado por varredura
# controlada (r=5..9, sem adjacencia): r=5 deixa speckle passar strict,
# r>=8 dilui o pico de celulas reais abaixo de 0.60; r=6..7 acertam os
# 3 slots. Escolhido o MENOR raio correto: r=6.
_SIGNAL_HALF = 6   # raio da janela de vizinhanca em pixels


def _build_signal_map(frame: np.ndarray) -> np.ndarray:
    """
    Pre-computa um mapa de sinal float32 [0, 1] para o frame inteiro.

    signal[y, x] = fracao de pixels na janela (2*HALF+1)^2 centrada em (x,y)
                   que indicam presenca de bloco.

    Usa cv2.boxFilter (implementacao C++) => muito mais rapido que
    calcular media de ROI por celula em Python puro.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s   = hsv[:, :, 1].astype(np.float32)
    v   = hsv[:, :, 2].astype(np.float32)

    colored_f    = ((s >= 80) & (v >= 45)).astype(np.float32)
    gray_block_f = ((s < 65)  & (v >= 110)).astype(np.float32)
    mask_f       = np.clip(colored_f + gray_block_f, 0.0, 1.0)

    color_f      = (s >= 55).astype(np.float32)
    bright_f     = ((s >= 35) & (v >= 50)).astype(np.float32)

    k = 2 * _SIGNAL_HALF + 1
    ksize = (k, k)
    sm  = cv2.boxFilter(mask_f,   -1, ksize)
    sc  = cv2.boxFilter(color_f,  -1, ksize)
    sb  = cv2.boxFilter(bright_f, -1, ksize)

    return sm * 0.55 + sc * 0.30 + sb * 0.15


def _lookup(sig: np.ndarray, x: float, y: float) -> float:
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < sig.shape[0] and 0 <= xi < sig.shape[1]:
        return float(sig[yi, xi])
    return 0.0


# ---------------------------------------------------------------------------
# Deteccao do tabuleiro
# ---------------------------------------------------------------------------

GRID_CANDIDATES = tuple(range(5, 13))


def _score_grid_n_grad(grad_proj: np.ndarray, n: int) -> float:
    """
    Pontua candidato n usando gradiente projetado.

    Para a grade correta, os separadores (bordas entre celulas) criam
    picos de gradiente consistentes em toda imagem — mesmo com celulas
    vazias. N incorreto coloca 'separadores' dentro de celulas, onde
    o gradiente e menor.

    Score = media do pico maximo de gradiente perto de cada separador.
    """
    size  = len(grad_proj)
    pitch = size / n
    sep_peaks = []
    for i in range(1, n):
        si   = int(i * pitch)
        half = max(2, int(pitch * 0.08))
        lo   = max(0, si - half)
        hi   = min(size, si + half + 1)
        sep_peaks.append(float(grad_proj[lo:hi].max()))
    return float(np.mean(sep_peaks)) if sep_peaks else 0.0


def detect_grid_size(frame: np.ndarray) -> tuple:
    """
    Detecta automaticamente quantas linhas e colunas tem o tabuleiro.

    Usa gradiente da imagem em cinza: bordas entre celulas geram picos
    consistentes de gradiente independente do estado (vazia/cheia).
    Para cada candidato n, pontua o maximo de gradiente proximo de cada
    separador esperado. O vencedor e o n cujos separadores mais se
    alinham com os picos reais da grade.
    """
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    crop = frame[gy1:gy2, gx1:gx2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Gradiente absoluto em cada direcao
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    abs_gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    # Projecao: media de gradiente por coluna e por linha
    col_proj = abs_gx.mean(axis=0)   # detecta separadores verticais
    row_proj = abs_gy.mean(axis=1)   # detecta separadores horizontais

    # Suavizacao leve para reduzir ruido de pixel sem apagar picos de borda
    col_proj = cv2.GaussianBlur(
        col_proj.reshape(1, -1), (1, 7), 1.5).reshape(-1)
    row_proj = cv2.GaussianBlur(
        row_proj.reshape(-1, 1), (7, 1), 1.5).reshape(-1)

    candidates = GRID_CANDIDATES
    col_scores = {n: _score_grid_n_grad(col_proj, n) for n in candidates}
    row_scores = {n: _score_grid_n_grad(row_proj, n) for n in candidates}

    best_cols = max(col_scores, key=col_scores.get)
    best_rows = max(row_scores, key=row_scores.get)

    def _confident(scores, best_n):
        vals = sorted(scores.values(), reverse=True)
        # Aceita se o melhor for ao menos 8% maior que o segundo
        return vals[0] > 0 and (vals[0] - vals[1]) / vals[0] >= 0.08

    row_confident = _confident(row_scores, best_rows)
    col_confident = _confident(col_scores, best_cols)
    if not row_confident or not col_confident or best_rows != best_cols:
        print(f"[grid] deteccao invalida: rows={best_rows} cols={best_cols} "
              f"row_conf={row_confident} col_conf={col_confident} "
              f"row_scores={dict(sorted(row_scores.items()))} "
              f"col_scores={dict(sorted(col_scores.items()))}")
        return None
    print(f"[grid] detectado {best_rows}x{best_cols}")
    return best_rows, best_cols


def detect_board(frame: np.ndarray, profile=None) -> Board:
    if profile is not None:
        # Geometria validada pelo catalogo: usa cantos+N do perfil,
        # sem presumir 8x8 e sem o teto de candidatos do auto-detect.
        rows = cols = int(profile["grid_size"])
        gx1, gy1 = profile["grid_top_left"]
        gx2, gy2 = profile["grid_bottom_right"]
    else:
        detected = detect_grid_size(frame)
        if detected is None:
            raise ValueError("board size was not detected with confidence")
        rows, cols = detected

        gx1, gy1 = config.GRID_TOP_LEFT
        gx2, gy2 = config.GRID_BOTTOM_RIGHT
    crop = frame[gy1:gy2, gx1:gx2]
    h, w = crop.shape[:2]

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _block_mask(hsv)

    cw   = w / cols
    ch   = h / rows
    radx = max(2, int(cw * 0.25))
    rady = max(2, int(ch * 0.25))

    board = Board(rows=rows, cols=cols)
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * cw)
            cy = int((r + 0.5) * ch)
            roi = mask[max(0, cy - rady):cy + rady + 1,
                       max(0, cx - radx):cx + radx + 1]
            if roi.size > 0 and np.mean(roi > 0) > 0.35:
                board.grid[r, c] = 1
                board.grid[r, c] = 1

    board.sync_bitboard()

    if config.DEBUG:
        total = rows * cols
        print(f"[board] {int(board.grid.sum())}/{total} cells filled ({rows}x{cols})")

    return board


# ---------------------------------------------------------------------------
# Deteccao de pecas (scan geometrico com mapa de sinal pre-computado)
# ---------------------------------------------------------------------------

def _normalize(cells) -> frozenset:
    """Normaliza celulas para origem (0,0)."""
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    mr, mc = min(rows), min(cols)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _snap_to_catalog(cells: list) -> list:
    """
    Encaixa o conjunto de celulas detectado na forma mais proxima do catalogo.

    - Se a deteccao ja coincide exatamente com uma entrada -> retorna ela.
    - Senao, calcula Jaccard (intersecao/uniao) com cada entrada.
    - Usa a entrada com maior Jaccard, desde que >= threshold.
    - Caso contrario, retorna None para forcar pausa segura.
    """
    if not cells:
        return None

    raw = _normalize(cells)
    n   = len(raw)

    # Rejeita ruído: peças reais devem estar no catálogo conhecido.
    if n < 1 or n > 9:
        return None

    best_j   = 0.0
    best_cat = None

    for cat in PIECE_CATALOG:
        diff = len(cat) - n
        if diff != 0:
            continue
        inter = len(raw & cat)
        union = len(raw | cat)
        j = inter / union if union else 0.0
        min_j = 0.75
        if j > best_j and j >= min_j:
            best_j   = j
            best_cat = cat

    if best_cat is not None:
        if best_cat != raw and config.DEBUG:
            print(f"  [snap] {sorted(raw)} -> {sorted(best_cat)} (J={best_j:.2f})")
        return sorted(best_cat)

    return None


def _max_run(b: np.ndarray) -> int:
    """Maior run de True consecutivos em qualquer linha ou coluna de b."""
    best = 0
    for mat in (b, b.T):
        for vec in mat:
            p    = np.concatenate([[0], vec.astype(np.int8), [0]])
            diff = np.diff(p)
            s    = np.where(diff == 1)[0]
            e    = np.where(diff == -1)[0]
            if s.size:
                best = max(best, int((e - s).max()))
    return best


def _is_native_scale(scale_x, scale_y, tol=0.01):
    """True se o frame está na resolução nativa 1080x2340.

    Portas de comportamento novo (ex.: peças 6-9 blocos, termos extras do
    score) valem no nativo — que é o único caso em produção (a captura
    normaliza tudo para 1080x2340). Fora do nativo vale o comportamento
    original, bit-idêntico.
    """
    return abs(scale_x - 1.0) < tol and abs(scale_y - 1.0) < tol


def score_candidate(mean_sig, support, n, run, empty_in_bbox, pitch,
                     scale_x, scale_y, nweak, jaccard):
    """Score único do candidato, compartilhado por detect_piece() e
    mirror_winner() (única fonte; os dois nunca mais divergem no score).

    Mesma expressão e mesma ordem de operações em todos os chamadores.
    nweak = nº de células com sinal < 0.75 (spill de grade errada); o
    termo fraco×pitch só vale em escala nativa.
    """
    pitch_err = abs(pitch - config.TRAY_CELL_PITCH * scale_x) / config.TRAY_CELL_PITCH
    native_frame = _is_native_scale(scale_x, scale_y)
    weak_pitch_term = 1.00 * nweak * pitch_err if native_frame else 0.0
    # Peso extra de pitch SOMENTE no nativo (alvo: grade comprimida 52
    # elegendo celula a mais; medido: fantasma-###@52 vs ## real@58 exige
    # peso efetivo >3.09; fora do nativo o peso base 1.0 continua valendo).
    pitch_w = 1.000 + (2.500 if native_frame else 0.0)
    return (mean_sig
            + 0.050 * support
            + 0.040 * n
            + 0.100 * run
            - 0.080 * empty_in_bbox
            - pitch_w * pitch_err
            - 0.15 * max(0, 5 - n) * (1.5 - mean_sig)  # prior anti-ruído cede a sinal saturado
            - weak_pitch_term  # spill so custa com grade errada (nativo)
            + 0.20 * jaccard)  # bônus por bater com catálogo


def detect_piece(sig: np.ndarray, slot_index: int,
                 rescue_maps=None) -> Optional[Piece]:
    """
    Detecta a geometria da peca no slot da bandeja.

    Vetorizado: faz lookup de todos os (dx, dy) de uma vez por pitch
    usando indexacao broadcast do NumPy, eliminando o loop Python de 36 chamadas.

    Score usa mean_sig (media, nao soma) para nao favorecer deteccoes maiores
    artificialmente. Barra de 5 ganha sobre barra de 4 quando ambas estao com
    sinal pleno — o que o range estendido de dx (ate ±half_pitch) garante.

    Suporta frames em qualquer escala: calcula escala a partir das dimensoes
    do frame recebido vs resolucao nativa do dispositivo.

    rescue_maps: tupla (pink_integral, gray_integral) de _rescue_maps() para
    o resgate localizado de diamante; None desativa o resgate.
    """
    sx_native, sy_native = config.TRAY_SLOT_CENTERS[slot_index]
    H, W   = sig.shape[:2]

    # Calcula escala do frame recebido vs resolucao nativa do dispositivo
    scale_x = W / NATIVE_DEVICE_W
    scale_y = H / NATIVE_DEVICE_H
    # Peças 6-9 blocos (ex.: quadrado 3x3) só no nativo; fora dele o
    # limite original 2-5 blocos continua valendo, bit-idêntico.
    n_max = 9 if _is_native_scale(scale_x, scale_y) else 5

    # Coordenadas escaladas para o frame atual
    sx = sx_native * scale_x
    sy = sy_native * scale_y

    # Pitch escalado para o frame atual. Faixa estreita em torno do pitch
    # fisico calibrado (58px): variacoes reais de escala/alinhamento passam,
    # mas pitch claramente incompativel (ex.: 50px comprimindo a grade e
    # criando celulas fantasmas) nem chega a ser avaliado.
    pitch_native = config.TRAY_CELL_PITCH
    pitch_min = 52.0 * scale_x
    pitch_max = 64.0 * scale_x
    pitch_step = 1.0 * scale_x

    best_score = -1.0
    best_b: Optional[np.ndarray] = None
    best_pitch = pitch_native
    best_dx, best_dy = 0.0, 0.0
    best_r0, best_c0 = 0, 0
    best_rescued: list = []

    cc_norm = np.arange(7, dtype=np.float64) - 3.0
    rr_norm = np.arange(7, dtype=np.float64) - 3.0

    # Busca de offsets restrita a ±8px do centro calibrado (passo 3px).
    # Experimento controlado provou que offsets livres (±~31px) geram
    # fantasmas nos 3 slots, enquanto a geometria calibrada acerta tudo:
    # o resíduo é overfit de posicionamento, não de pitch nem de score.
    _DXDY_LIM = 8.0
    for pitch in np.arange(pitch_min, pitch_max, pitch_step):
        half    = int(pitch / 2) + 2
        # union1d garante que o centro calibrado (0,0) seja sempre avaliado:
        # o passo 3 a partir de -8 nunca atinge o zero sozinho.
        dx_arr  = np.union1d(
            np.arange(max(-half, -_DXDY_LIM), min(half, _DXDY_LIM) + 1,
                      3, dtype=np.float64),
            np.array([0.0]),
        )
        dy_arr  = np.union1d(
            np.arange(max(-half, -_DXDY_LIM), min(half, _DXDY_LIM) + 1,
                      3, dtype=np.float64),
            np.array([0.0]),
        )

        # Coordenadas pixel para cada (dx, cc) e (dy, rr): shape (D, 7)
        all_xi_f = sx + dx_arr[:, None] + cc_norm[None, :] * pitch
        all_yi_f = sy + dy_arr[:, None] + rr_norm[None, :] * pitch
        all_xi   = np.round(all_xi_f).astype(np.int32)
        all_yi   = np.round(all_yi_f).astype(np.int32)

        x_oob = (all_xi < 0) | (all_xi >= W)
        y_oob = (all_yi < 0) | (all_yi >= H)
        xi_c  = np.clip(all_xi, 0, W - 1)
        yi_c  = np.clip(all_yi, 0, H - 1)

        # Lookup batch: vals_all[i_dx, j_dy, rr, cc] = sig[yi[j_dy,rr], xi[i_dx,cc]]
        vals_all = sig[
            yi_c[np.newaxis, :, :, np.newaxis],
            xi_c[:, np.newaxis, np.newaxis, :],
        ].astype(np.float64)

        oob = (x_oob[:, np.newaxis, np.newaxis, :]
               | y_oob[np.newaxis, :, :, np.newaxis])
        vals_all[oob] = 0.0

        # Threshold estrito, SEM expansao por adjacencia: a adjacencia
        # (vals>=0.40 + vizinho strict) foi removida porque o diagnostico
        # provou que ela religa brilho espalhado pelo blur como celula
        # (ex.: 0.48/0.43 -> fantasmas), enquanto o blur r=6 ja preserva os
        # picos reais acima de 0.60 sem precisar de resgate.
        strict = vals_all >= 0.60
        bits_all = strict

        # Itera apenas sobre pares com alguma celula detectada
        for idx in np.argwhere(bits_all.any(axis=(-1, -2))):
            i_dx, j_dy = int(idx[0]), int(idx[1])
            bits = bits_all[i_dx, j_dy]
            vals = vals_all[i_dx, j_dy]

            rows_w = np.where(bits.any(axis=1))[0]
            cols_w = np.where(bits.any(axis=0))[0]
            if rows_w.size == 0 or cols_w.size == 0:
                continue
            r0, r1 = int(rows_w[0]),  int(rows_w[-1]) + 1
            c0, c1 = int(cols_w[0]), int(cols_w[-1]) + 1

            n_cells = int(bits[r0:r1, c0:c1].sum())
            if n_cells < 1 or n_cells > n_max:
                continue

            # Não recorta linhas/cols com blocos - mantém bbox original da detecção
            # O recorte agressivo estava transformando 2x3 em 2x2

            b  = bits[r0:r1, c0:c1]
            bv = vals[r0:r1, c0:c1]
            n  = int(b.sum())
            if n < 1 or n > n_max:
                continue

            mean_sig = float(bv[b].mean())
            # Rejeita sinal fraco (ruído isolado) - MAIS ALTO
            if mean_sig < 0.55:
                continue

            bp       = np.pad(b.astype(np.int8), 1)
            nbrs     = (bp[:-2, 1:-1] + bp[2:, 1:-1] +
                        bp[1:-1, :-2] + bp[1:-1, 2:])
            support  = int(((nbrs > 0) & b).sum())

            run = _max_run(b)
            bh2, bw2 = r1 - r0, c1 - c0
            empty_in_bbox = bh2 * bw2 - n

            # Verifica se a forma detectada bate com catálogo ANTES de pontuar
            detected_cells = [(r, c) for r in range(bh2) for c in range(bw2) if b[r, c]]
            snapped = _snap_to_catalog(detected_cells)
            if snapped is None or len(snapped) != n:
                # Unknown geometry is not safe to send to the solver.
                continue
            # Calcula Jaccard com a forma snapped
            raw_norm = _normalize(detected_cells)
            snap_norm = _normalize(snapped)
            jaccard = len(raw_norm & snap_norm) / len(raw_norm | snap_norm) if (raw_norm | snap_norm) else 0
            if jaccard < 0.70:
                # Não bate bem com nenhuma peça conhecida
                continue

            # Peso do pitch comensuravel com os bonus de forma: um pitch 6px
            # fora do calibrado custa ~0.10, o suficiente para que uma celula
            # fantasma (+~0.21 em n/run/tamanho) nao compense a geometria errada.
            # Variacao real pequena (±2px) custa <=0.035: sem vies excessivo.
            # Interacao fraca×pitch SOMENTE em escala nativa: celulas fracas
            # so penalizam quando a grade esta errada (spill de pitch
            # incorreto). Em frames escalados o blur achata o sinal e o
            # termo perderia poder discriminativo — la, vale o score base.
            # Medido em 2026-09-05 (1080x2340): fantasmas nweak=2-3,
            # err~0.10; formas reais nweak=0-1, err~0.
            score = score_candidate(mean_sig, support, n, run,
                                    empty_in_bbox, pitch, scale_x, scale_y,
                                    int((bv[b] < 0.75).sum()), jaccard)

            if score > best_score:
                best_score = score
                best_b     = b.copy()
                best_pitch = pitch
                best_dx    = float(dx_arr[i_dx])
                best_dy    = float(dy_arr[j_dy])
                best_r0, best_c0 = r0, c0

    if best_b is None or best_b.sum() == 0:
        return None

    # Score mínimo para aceitar a detecção (evita ruído/ghost pieces)
    if best_score < 0.75:
        if config.DEBUG:
            print(f"  [reject] slot {slot_index}: best_score={best_score:.3f} < 0.75")
        return None

    # Completude localizada de diamante sobre o VENCEDOR (mesmo pitch e
    # mesma grade do vencedor): adiciona celula 0-bit vizinha somente com
    # assinatura de diamante confirmada. Nao re-pontua e nao muda a busca.
    if rescue_maps is not None:
        comp = _complete_winner_with_diamonds(
            sig, W, H, sx, sy, best_pitch, best_dx, best_dy,
            best_r0, best_c0, best_b, slot_index, rescue_maps)
        if comp is not None:
            best_b, best_r0, best_c0, best_rescued = comp

    bh, bw = best_b.shape

    mask5 = np.zeros((5, 5), dtype=np.uint8)
    r_off = (5 - bh) // 2
    c_off = (5 - bw) // 2
    mask5[r_off:r_off + bh, c_off:c_off + bw] = best_b.astype(np.uint8)

    piece = Piece.from_mask(piece_id=slot_index, mask=mask5)
    piece.tray_cx = float(sx)
    piece.tray_cy = float(sy)

    # Encaixa no catalogo de formas conhecidas - validação final
    snapped_cells = _snap_to_catalog(piece.cells)
    if snapped_cells is None:
        return None
    if snapped_cells != piece.cells:
        piece2 = Piece(
            id=slot_index,
            cells=snapped_cells,
            width=max(c for _, c in snapped_cells) + 1 if snapped_cells else 0,
            height=max(r for r, _ in snapped_cells) + 1 if snapped_cells else 0,
        )
        piece2.tray_cx = float(sx)
        piece2.tray_cy = float(sy)
        piece = piece2

    # Validação final: peça deve permanecer numa forma catalogada.
    if len(piece.cells) < 1 or len(piece.cells) > n_max:
        if config.DEBUG:
            print(f"  [reject] slot {slot_index}: final piece has {len(piece.cells)} blocks")
        return None

    # Diamantes resgatados, nas coordenadas finais da peca (somente as que
    # sobreviveram ao snap; as demais sao descartadas com a forma).
    final_set = set(piece.cells)
    piece.rescued_cells = [c for c in best_rescued if c in final_set]
    if piece.rescued_cells and config.DEBUG:
        print(f"  [rescue] slot {slot_index}: {len(piece.rescued_cells)} celula(s) "
              f"com diamante em {piece.rescued_cells}")

    if config.DEBUG:
        print(f"PECA {slot_index}: {piece.width}x{piece.height}, "
              f"blocos={len(piece.cells)}, pitch={best_pitch:.1f}, score={best_score:.3f}")
        grid = [['.' for _ in range(piece.width)] for _ in range(piece.height)]
        for r, c in piece.cells:
            if 0 <= r < piece.height and 0 <= c < piece.width:
                grid[r][c] = '#'
        for row in grid:
            print("  " + "".join(row))

    # Centroide detectado dos blocos em pixels do frame (para o pickup).
    # Recalculado da grade vencedora (mesma fórmula da busca); independe
    # do snap (usa as células reais detectadas, não a forma ajustada).
    _wx = np.round(sx + best_dx + cc_norm * best_pitch).astype(int)
    _wy = np.round(sy + best_dy + rr_norm * best_pitch).astype(int)
    _occ = [(int(np.clip(_wx[best_c0 + c], 0, W - 1)),
             int(np.clip(_wy[best_r0 + r], 0, H - 1)))
            for r in range(best_b.shape[0]) for c in range(best_b.shape[1])
            if best_b[r, c]]
    if _occ:
        piece.pickup_cx = float(sum(p[0] for p in _occ) / len(_occ))
        piece.pickup_cy = float(sum(p[1] for p in _occ) / len(_occ))

    return piece


def read_pieces(frame: np.ndarray) -> List[Optional[Piece]]:
    sig = _build_signal_map(frame)
    maps = _rescue_maps(frame)
    return [detect_piece(sig, i, rescue_maps=maps) for i in range(3)]


# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------

def draw_overlay(frame: np.ndarray,
                 board: Board,
                 pieces: List[Optional[Piece]]) -> np.ndarray:
    vis = frame.copy()
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    bw = gx2 - gx1
    bh = gy2 - gy1
    rows, cols = board.rows, board.cols
    px = bw / cols
    py = bh / rows

    for i in range(cols + 1):
        cv2.line(vis, (int(gx1 + i * px), gy1), (int(gx1 + i * px), gy2), (80, 80, 80), 1)
    for i in range(rows + 1):
        cv2.line(vis, (gx1, int(gy1 + i * py)), (gx2, int(gy1 + i * py)), (80, 80, 80), 1)

    for r in range(rows):
        for c in range(cols):
            cx = int(gx1 + (c + .5) * px)
            cy = int(gy1 + (r + .5) * py)
            color = (0, 255, 0) if board.grid[r, c] else (50, 50, 50)
            cv2.circle(vis, (cx, cy), 8, color, -1)

    half = config.PIECE_SLOT_HALF
    for i, (sx, sy) in enumerate(config.TRAY_SLOT_CENTERS):
        cv2.rectangle(vis, (sx - half, sy - half), (sx + half, sy + half),
                      (255, 160, 0), 2)
        p = pieces[i] if i < len(pieces) else None
        label = f"S{i}: {p.width}x{p.height} [{len(p.cells)}blk]" if p else f"S{i}: vazio"
        cv2.putText(vis, label, (sx - half + 4, sy + half - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        if p and p.cells:
            # Desenha pontos proporcionais (como no tabuleiro) em vez de mini-retângulos
            filled = set(p.cells)
            grid_h = max(r for r, _ in p.cells) + 1
            grid_w = max(c for _, c in p.cells) + 1

            # Escala: usa o pitch da bandeja para calcular tamanho proporcional
            cell_pitch = config.TRAY_CELL_PITCH
            dot_radius = max(3, int(cell_pitch * 0.18))  # ~10px para pitch 58

            # Centraliza a peça no slot
            total_w = (grid_w - 1) * cell_pitch
            total_h = (grid_h - 1) * cell_pitch
            ox = sx - total_w / 2.0
            oy = sy - total_h / 2.0 - half + 10  # ligeiramente acima do centro

            for r, c in filled:
                cx = int(ox + c * cell_pitch)
                cy = int(oy + r * cell_pitch)
                # Desenha ponto (como no tabuleiro)
                cv2.circle(vis, (cx, cy), dot_radius, (0, 200, 255), -1)
                cv2.circle(vis, (cx, cy), dot_radius + 1, (0, 255, 255), 1)

    return vis


def draw_move(vis: np.ndarray, move, piece: Piece, board: Board = None) -> np.ndarray:
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    if board is not None:
        px = (gx2 - gx1) / board.cols
        py = (gy2 - gy1) / board.rows
    else:
        px, py = config.CELL_W, config.CELL_H
    for dr, dc in piece.cells:
        r, c = move.row + dr, move.col + dc
        cx = int(gx1 + (c + .5) * px)
        cy = int(gy1 + (r + .5) * py)
        cv2.circle(vis, (cx, cy), int(px * 0.35), (0, 220, 255), 3)
    return vis
