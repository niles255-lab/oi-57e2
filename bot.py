"""
BlockBlaster Bot  |  Redmi Note 8 / Blocks of Bitcoin (Fumb Games)

Uso:
  python bot.py           -- auto-play imediato
  python bot.py --watch   -- apenas observar, sem jogar

Teclas:
  A     -- alternar auto-play / observacao
  H     -- segurar peca e capturar overlay automaticamente
  [/]   -- DRAG_LIFT_Y -10 / +10 px
  ,/.   -- DRAG_ROW_OFFSET -0.5 / +0.5 celulas
  S     -- salvar DRAG_LIFT_Y e DRAG_ROW_OFFSET em config.py
  I     -- salvar frame + mask + overlay de diagnostico
  Q/ESC -- sair
"""

import sys
import time
import math
import datetime
import threading
import pathlib

import cv2
import numpy as np

import config
import adb_io
import board_catalog
from vision import detect_board, read_pieces, draw_overlay, draw_move, _block_mask
from model import Board, Piece
from model import is_legal, apply_move
from solver import best_sequence


# ---------------------------------------------------------------------------
# Logging — espelha stdout para arquivo e mantém pasta da sessão
# ---------------------------------------------------------------------------

class _Tee:
    """Escreve simultaneamente em stdout e em um arquivo de texto."""
    def __init__(self, path: pathlib.Path):
        self._orig = sys.stdout
        self._f    = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._orig.write(data)
        self._f.write(data)

    def flush(self):
        self._orig.flush()
        self._f.flush()

    def close(self):
        sys.stdout = self._orig
        self._f.close()

    # needed for cv2 / subprocess que checam isatty
    def isatty(self):
        return False


def _init_session_dir() -> pathlib.Path:
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d   = pathlib.Path(__file__).parent / "logs" / f"run_{ts}"
    (d / "diag").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Captura via scrcpy stream (substitui Capturer thread + cap_q)
# ---------------------------------------------------------------------------
# A captura agora é síncrona via adb_io.capture_frame() que usa scrcpy stream
# com fallback para ADB screencap. Não precisa mais de thread Capturer + queue.


# ---------------------------------------------------------------------------
# Execucao de jogada
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Geometria ativa do tabuleiro (perfil validado pelo board_catalog).
# Nenhuma jogada ADB pode acontecer sem um perfil válido aqui: a porta
# _board_gate só libera o autoplay junto com um perfil confirmado.
# ---------------------------------------------------------------------------
_BOARD_PROFILE = None


def _profile_geom(board=None):
    """(gx1, gy1, cw, ch) do perfil validado, ou None (fallback config).

    Só usa o perfil quando as dimensões do board conferem com ele;
    caso contrário cai no comportamento legado (display).
    """
    p = _BOARD_PROFILE
    if p is not None and board is not None:
        try:
            n = int(p["grid_size"])
        except (KeyError, TypeError, ValueError):
            return None
        if board.rows == n and board.cols == n:
            (x1, y1), (x2, y2) = p["grid_top_left"], p["grid_bottom_right"]
            return (x1, y1), (x2 - x1) / n, (y2 - y1) / n
    return None


def _cell_size(board=None):
    g = _profile_geom(board)
    if g is not None:
        return g[1], g[2]
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    if board is not None:
        return (gx2 - gx1) / board.cols, (gy2 - gy1) / board.rows
    return config.CELL_W, config.CELL_H


def _cell_px(row: float, col: float, board=None):
    g = _profile_geom(board)
    if g is not None:
        (gx1, gy1), cw, ch = g
    else:
        gx1, gy1 = config.GRID_TOP_LEFT
        cw, ch = _cell_size(board)
    return (
        int(gx1 + (col + 0.5) * cw),
        int(gy1 + (row + 0.5) * ch),
    )


def _target_pixel(move, piece, board=None) -> tuple:
    """Return the pixel centroid of the cells the game must occupy."""
    g = _profile_geom(board)
    if g is not None:
        (gx1, gy1), cw, ch = g
    else:
        gx1, gy1 = config.GRID_TOP_LEFT
        cw, ch = _cell_size(board)
    if not piece.cells:
        raise ValueError("cannot calculate a target for an empty piece")
    mean_r = move.row + sum(r for r, _ in piece.cells) / len(piece.cells)
    mean_c = move.col + sum(c for _, c in piece.cells) / len(piece.cells)
    return (
        int(round(gx1 + (mean_c + 0.5) * cw)),
        int(round(gy1 + (mean_r + 0.5) * ch + config.DRAG_LIFT_Y)),
    )


def _target_ey(move, piece, board=None) -> tuple:
    """Compatibility alias for diagnostics using the old helper name."""
    return _target_pixel(move, piece, board)


def _visual_offset(held: np.ndarray, before: np.ndarray,
                   board=None, finger=(0, 0)):
    """Deslocamento visual-dedo medido por diff (somente leitura).

    Retorna (ox, oy, ok): centroide da massa que mudou dentro do
    retangulo do tabuleiro menos a posicao do dedo. ok=False quando
    sem massa suficiente ou offset absurdo (>3 celulas) — ai o
    chamador solta sem correcao. Nunca mexe em nada.
    """
    try:
        if held is None or before is None or held.shape != before.shape:
            return 0, 0, False
        g = _profile_geom(board)
        if g is not None:
            (gx1, gy1), cw, ch = g
            n = board.rows
            x1, y1 = int(gx1), int(gy1)
            x2, y2 = int(gx1 + n * cw), int(gy1 + n * ch)
            cell = max(cw, ch)
        else:
            x1, y1 = config.GRID_TOP_LEFT
            x2, y2 = config.GRID_BOTTOM_RIGHT
            cell = max(config.CELL_W, config.CELL_H)
        H, W = held.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, False
        hb = cv2.cvtColor(held, cv2.COLOR_BGR2GRAY).astype(float)
        bb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(float)
        d = np.abs(hb[y1:y2, x1:x2] - bb[y1:y2, x1:x2])
        m = np.zeros_like(d)
        m[d > 30.0] = d[d > 30.0]
        if int((d > 30.0).sum()) < 500:
            return 0, 0, False
        tot = m.sum()
        ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        vx, vy = x1 + (xs * m).sum() / tot, y1 + (ys * m).sum() / tot
        ox, oy = vx - finger[0], vy - finger[1]
        if math.hypot(ox, oy) > 3 * cell:
            return 0, 0, False
        return int(round(ox)), int(round(oy)), True
    except Exception:
        return 0, 0, False


def _ghost_visible(held: np.ndarray, before: np.ndarray,
                   move, piece: Piece, board=None) -> bool:
    """Verifica se o ghost da peca esta visivel nas celulas alvo."""
    g = _profile_geom(board)
    if g is not None:
        (gx1, gy1), _, _ = g
    else:
        gx1, gy1 = config.GRID_TOP_LEFT
    cw, ch = _cell_size(board)
    max_r = board.rows if board is not None else 8
    max_c = board.cols if board is not None else 8
    diffs = []
    for r, c in piece.cells:
        tr, tc = move.row + r, move.col + c
        if tr >= max_r or tc >= max_c:
            return False
        cx = int(gx1 + (tc + 0.5) * cw)
        cy = int(gy1 + (tr + 0.5) * ch)
        rad = int(ch * 0.28)
        b = before[max(0, cy - rad):cy + rad, max(0, cx - rad):cx + rad]
        h = held[max(0, cy - rad):cy + rad, max(0, cx - rad):cx + rad]
        if b.size == 0 or h.size == 0:
            continue
        diffs.append(float(cv2.absdiff(b, h).mean()))
    if not diffs:
        return False
    return (sum(diffs) / len(diffs)) > 6.0


def execute_move(move, piece: Piece, before_frame: np.ndarray = None, board=None) -> bool:
    sx, sy = config.TRAY_SLOT_CENTERS[move.piece_index]
    # Pickup no centroide detectado da peca (robusto a calibragem velha e
    # jitter); fallback = centro do slot quando indisponivel/duvidoso.
    pcx = getattr(piece, "pickup_cx", None)
    pcy = getattr(piece, "pickup_cy", None)
    if (pcx is not None and pcy is not None
            and abs(pcx - sx) <= 64 and abs(pcy - sy) <= 64):
        px, py = int(pcx), int(pcy)
        print(f"[PICKUP]\nslot={move.piece_index}\n"
              f"detected_centroid=({px},{py})\n"
              f"configured_center=({sx},{sy})\n"
              f"touch_down=({px},{py})\nsource=centroid")
    else:
        px, py = sx, sy
        print(f"[PICKUP]\nslot={move.piece_index}\n"
              f"touch_down=({px},{py})\nsource=slot-center-fallback")
    ex, ey = _target_pixel(move, piece, board)

    # The finger and the visual piece share the same measured centroid. If a
    # fallback pickup is used, preserve the measured source-to-finger offset.
    source_visual = (
        (float(piece.pickup_cx), float(piece.pickup_cy))
        if piece.pickup_cx is not None and piece.pickup_cy is not None
        else (float(sx), float(sy))
    )
    grab_offset = (source_visual[0] - px, source_visual[1] - py)
    cx = int(round(ex - grab_offset[0]))
    cy = int(round(ey - grab_offset[1]))
    print("[TRANSPORT] continuous ADB touchscreen gesture")
    print(f"[SWIPE] ({px},{py}) -> ({cx},{cy}) "
          f"dur={config.DRAG_DURATION_MS}ms target_visual=({ex},{ey}) "
          f"grab_offset=({grab_offset[0]:.1f},{grab_offset[1]:.1f})")
    return adb_io.swipe(px, py, cx, cy, config.DRAG_DURATION_MS)


# ---------------------------------------------------------------------------
# Persistencia de configuracao
# ---------------------------------------------------------------------------

def _save_lift_y(value: int):
    import re, pathlib
    path     = pathlib.Path(__file__).parent / "config.py"
    text     = path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"^(DRAG_LIFT_Y\s*:\s*int\s*=\s*)-?\d+",
        rf"\g<1>{value}", text, flags=re.MULTILINE,
    )
    if new_text == text:
        new_text = re.sub(
            r"^(DRAG_LIFT_Y\s*=\s*)-?\d+",
            rf"\g<1>{value}", text, flags=re.MULTILINE,
        )
    path.write_text(new_text, encoding="utf-8")
    print(f"[cfg] DRAG_LIFT_Y = {value} salvo em config.py")


def _save_row_offset(value: float):
    import re, pathlib
    path     = pathlib.Path(__file__).parent / "config.py"
    text     = path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"^(DRAG_ROW_OFFSET\s*:\s*float\s*=\s*)[\d.]+",
        rf"\g<1>{value}", text, flags=re.MULTILINE,
    )
    if new_text == text:
        new_text = re.sub(
            r"^(DRAG_ROW_OFFSET\s*=\s*)[\d.]+",
            rf"\g<1>{value}", text, flags=re.MULTILINE,
        )
    path.write_text(new_text, encoding="utf-8")
    print(f"[cfg] DRAG_ROW_OFFSET = {value} salvo em config.py")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_WIN = "BlockBlaster Bot"


def show(frame, board, pieces, seq, auto_play, move_count, status=""):
    vis = draw_overlay(frame, board, pieces)

    # Mostra apenas a PRIMEIRA jogada da sequencia — e so se for legal
    if seq:
        move  = seq[0]
        piece = next((p for p in pieces
                      if p is not None and p.id == move.piece_index), None)
        if piece is not None and is_legal(board, piece, move.row, move.col):
            vis = draw_move(vis, move, piece, board)

    label = status if status else ("AUTO-PLAY" if auto_play else "OBSERVANDO")
    color = (0, 200, 255) if status else ((0, 255, 80) if auto_play else (0, 165, 255))
    cv2.putText(vis, label,              (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Jogadas: {move_count}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, f"LiftY={config.DRAG_LIFT_Y}  RowOff={config.DRAG_ROW_OFFSET}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 255), 1, cv2.LINE_AA)

    scale = min(900 / vis.shape[1], 1100 / vis.shape[0])
    cv2.imshow(_WIN, cv2.resize(vis, None, fx=scale, fy=scale))


# ---------------------------------------------------------------------------
# Verificacao pos-jogada + protecao contra estagnacao (somente bot.py)
# ---------------------------------------------------------------------------

def _board_sig(board) -> tuple:
    """Assinatura comparavel do tabuleiro (dimensoes + bytes da grade)."""
    if board is None:
        return (-1, -1, b"")
    return (board.rows, board.cols, board.grid.tobytes())


def _pieces_sig(pieces) -> tuple:
    """Assinatura comparavel da bandeja (slot, tamanho e celulas)."""
    if not pieces:
        return ()
    return tuple(sorted(
        (p.id, p.width, p.height, tuple(sorted(p.cells)))
        for p in pieces if p is not None and p.cells
    ))


def _state_sig(board, pieces) -> tuple:
    return (_board_sig(board), _pieces_sig(pieces))


def _fmt_board(board) -> str:
    if board is None:
        return "  <sem tabuleiro>"
    lines = [f"  {board.rows}x{board.cols} fill={int(board.grid.sum())}"]
    for r in range(board.rows):
        lines.append("  " + "".join("#" if board.grid[r, c] else "."
                                    for c in range(board.cols)))
    return "\n".join(lines)


def _fmt_pieces(pieces) -> str:
    if not pieces:
        return "  <sem pecas>"
    lines = []
    for p in pieces:
        if p is None or not p.cells:
            lines.append("  slot ?: vazio")
        else:
            lines.append(f"  slot {p.id}: {p.width}x{p.height} "
                         f"[{len(p.cells)}b] {sorted(p.cells)}")
    return "\n".join(lines)


def _board_gate(frame, active):
    """Porta do catalogo de grades no autoplay (usa a MESMA frame).

    Retorna (auto_play, profile). Sem moves sem perfil validado:
    - conhecida compativel: (True, perfil salvo) em silencio (logs so ao
      estabelecer/trocar de perfil).
    - nova / variante / sem confianca: mostra dados, pergunta [s/N].
      's' salva+registra e retoma com o perfil novo; 'n'/EOF pausa
      (A=retomar manual). Nunca altera detector, solver, pecas ou captura.
    """
    status, det, saved = board_catalog.ensure_board(frame)
    if status == "known":
        if active is None or active.get("shape") != saved.get("shape"):
            print(f"[BOARD] Detectado: {saved['shape']}")
            print(f"[BOARD] Perfil: {saved['shape']}")
            print("[BOARD] Geometria validada")
            print("[AUTOPLAY] Board pronto")
            print("[AUTOPLAY] Liberando detector/solver")
        return True, saved
    if status == "unknown":
        print("[board] AUTOPLAY -> PAUSADO | motivo: grade sem confianca "
              "para jogada segura (nada salvo, nada movido)")
        return False, active
    if status == "variant":
        n = det["grid_size"]
        print(f"[board] AUTOPLAY -> PAUSADO | motivo: {n}x{n} conhecida, "
              f"mas geometria diferente da cadastrada")
        print(f"  cadastrada: cantos {saved['grid_top_left']}->"
              f"{saved['grid_bottom_right']} pitch {saved['pitch_x']}")
        print(f"  detectada : cantos {det['grid_top_left']}->"
              f"{det['grid_bottom_right']} pitch {det['pitch_x']}")
    else:  # new
        n = det["grid_size"]
        print("NOVA GRADE DETECTADA")
        print(f"Tamanho: {n}x{n}")
        print(f"Resolucao: {det['frame_size'][0]}x{det['frame_size'][1]}")
        print(f"Top-left: {det['grid_top_left']}")
        print(f"Bottom-right: {det['grid_bottom_right']}")
        area = det["area_px"]
        print(f"Area: {area[0]}x{area[1]}")
        print(f"Pitch: {det['pitch_x']} x {det['pitch_y']}")
        print(f"Confianca: cols {det['col_margin_pct']}% / "
              f"linhas {det['row_margin_pct']}%")
        print(board_catalog.ascii_map(n))
        print(f"[board] AUTOPLAY -> PAUSADO | motivo: grade {n}x{n} nova "
              f"(nenhuma jogada executada)")
    try:
        if status == "variant":
            ans = input(f"Salvar nova configuracao {det['grid_size']}x"
                        f"{det['grid_size']} e continuar? [s/N] ")
        else:
            ans = input(f"Nova grade {det['grid_size']}x{det['grid_size']} "
                        f"detectada. Salvar no catalogo e continuar? [s/N] ")
    except EOFError:
        ans = "n"
    if ans.strip().lower() != "s":
        print("[board] grade NAO catalogada — pausado (A=retomar manual, "
              "Q=sair)")
        return False, active
    d, _ = board_catalog.save_profile(det, frame)
    print(f"[board] CATALOGACAO -> perfil {det['shape']} salvo em {d} "
          f"-> AUTOPLAY (retomando sozinho)")
    print(f"[BOARD] Detectado: {det['shape']}")
    print(f"[BOARD] Perfil: {det['shape']}")
    print("[BOARD] Geometria validada")
    print("[AUTOPLAY] Board pronto")
    print("[AUTOPLAY] Liberando detector/solver")
    return True, det


def verify_post_move(before_board, before_pieces, move, profile=None):
    """Rele o estado real e compara com o anterior a jogada.

    Retorna (verdict, after_board, after_pieces) com verdict em
    {"EFFECTIVE", "NOOP", "WRONG_PLACEMENT", "DROP_FAILURE",
     "VERIFICATION_FAILURE"}.
    EFFECTIVE exige board completo esperado e peça removida da bandeja.
    """
    after_frame = adb_io.capture_frame()
    if after_frame is None:
        return "UNKNOWN", None, None
    after_board = detect_board(after_frame, profile=profile)
    after_pieces = read_pieces(after_frame)

    def slot_sig(plist, idx):
        for p in (plist or []):
            if p is not None and p.id == idx and p.cells:
                return (p.width, p.height, tuple(sorted(p.cells)))
        return None

    tray_changed = (slot_sig(after_pieces, move.piece_index)
                    != slot_sig(before_pieces, move.piece_index))
    board_changed = (_board_sig(after_board) != _board_sig(before_board))

    _missing: list = []
    _extra: list = []
    try:
        _p = next((x for x in (before_pieces or [])
                   if x is not None and x.id == move.piece_index and x.cells),
                  None)
        if _p is None:
            return "VERIFICATION_FAILURE", after_board, after_pieces
        import numpy as _np
        _exp, _, _ = apply_move(before_board, _p, move.row, move.col)
        _exp_set = set(zip(*_np.where(_exp.grid == 1)))
        _aft_set = set(zip(*_np.where(after_board.grid == 1)))
        _missing = sorted(_exp_set - _aft_set)
        _extra = sorted(_aft_set - _exp_set)
        if _missing or _extra:
            print(f"[MOVE ERROR] expected-full-missing={_missing} "
                  f"actual-extra={_extra}")
    except Exception as _e:
        print(f"[MOVE ERROR] reason=verify-diagnostic-failed ({_e})")
        return "VERIFICATION_FAILURE", after_board, after_pieces

    if board_changed and tray_changed and not _missing and not _extra:
        return "EFFECTIVE", after_board, after_pieces
    if board_changed:
        return "WRONG_PLACEMENT", after_board, after_pieces
    if tray_changed:
        return "DROP_FAILURE", after_board, after_pieces
    return "NOOP", after_board, after_pieces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    session_dir = _init_session_dir()
    tee = _Tee(session_dir / "log.txt")
    sys.stdout = tee

    auto_play = "--watch" not in sys.argv

    print("=" * 55)
    print("BlockBlaster Bot  |  Redmi Note 8 / Blocks of Bitcoin")
    print("=" * 55)
    print(f"Session: {session_dir}")
    print(f"Device : {config.ADB_DEVICE_ID}")
    print(f"Modo   : {'AUTO-PLAY' if auto_play else 'OBSERVACAO (--watch)'}")
    print(f"LiftY    : {config.DRAG_LIFT_Y}px")
    print(f"RowOffset: {config.DRAG_ROW_OFFSET} celulas")
    print("Teclas : A=auto  H=hold  [/]=LiftY  ,/.=RowOffset  S=salvar  I=diag  Q=sair")
    print()

    if not adb_io.check_device():
        print("ERRO: dispositivo ADB nao encontrado.")
        sys.exit(1)
    print("[OK] dispositivo conectado\n")

    # Inicializa scrcpy stream (novo - substitui Capturer thread + cap_q)
    print("[scrcpy] iniciando stream de vídeo...")
    if adb_io.init_scrcpy_stream():
        print("[scrcpy] OK — stream de vídeo ativo")
    else:
        print("[scrcpy] FALHOU — usando ADB screencap como fallback")

    frame      = None
    board      = Board()
    pieces     = [None, None, None]
    seq        = []
    move_count = 0
    fail_streak = 0
    MAX_FAILS   = 4
    no_move_streak = 0
    # Perfil de grade validado (None = nenhum: nenhuma jogada ADB acontece).
    active_profile = None
    # Protecao contra estagnacao: conta verificacoes NOOP seguidas sobre o
    # MESMO estado (tabuleiro+bandeja). Ao atingir o limite, para de executar
    # e so volta a jogar quando o estado mudar de verdade.
    prev_sig   = None
    stagnant_count = 0
    STAGNANT_LIMIT = 3
    observing_only = False
    stable_state_sig = None
    stable_state_count = 0

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    print("[aguardando primeiro frame do stream...]")

    while True:

        # ----------------------------------------------------------------
        # Teclado + display a ~30 fps
        # ----------------------------------------------------------------
        key = cv2.waitKey(33) & 0xFF

        if key in (ord('q'), 27):
            break

        if key == ord('a'):
            auto_play = not auto_play
            print(f"[modo] {'AUTO-PLAY' if auto_play else 'OBSERVACAO'}")

        if key == ord('['):
            config.DRAG_LIFT_Y -= 10
            print(f"[cfg] DRAG_LIFT_Y = {config.DRAG_LIFT_Y}")

        if key == ord(']'):
            config.DRAG_LIFT_Y += 10
            print(f"[cfg] DRAG_LIFT_Y = {config.DRAG_LIFT_Y}")

        if key == ord(','):
            config.DRAG_ROW_OFFSET = round(config.DRAG_ROW_OFFSET - 0.5, 1)
            print(f"[cfg] DRAG_ROW_OFFSET = {config.DRAG_ROW_OFFSET}")

        if key == ord('.'):
            config.DRAG_ROW_OFFSET = round(config.DRAG_ROW_OFFSET + 0.5, 1)
            print(f"[cfg] DRAG_ROW_OFFSET = {config.DRAG_ROW_OFFSET}")

        if key == ord('s'):
            _save_lift_y(config.DRAG_LIFT_Y)
            _save_row_offset(config.DRAG_ROW_OFFSET)

        if key == ord('i') and frame is not None:
            ts   = datetime.datetime.now().strftime("%H%M%S")
            gx1, gy1 = config.GRID_TOP_LEFT
            gx2, gy2 = config.GRID_BOTTOM_RIGHT
            hsv  = cv2.cvtColor(frame[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2HSV)
            d    = session_dir / "diag"
            cv2.imwrite(str(d / f"frame_{ts}.png"),   frame)
            cv2.imwrite(str(d / f"mask_{ts}.png"),    _block_mask(hsv))
            cv2.imwrite(str(d / f"overlay_{ts}.png"),
                        cv2.resize(draw_overlay(frame, board, pieces),
                                   None, fx=0.5, fy=0.5))
            print(f"[diag] {d / f'frame_{ts}.png'}  mask  overlay")

        if key == ord('h'):
            slot = next(
                (i for i, p in enumerate(pieces) if p is not None and p.cells),
                None,
            )
            if slot is None:
                print("[hold] nenhuma peca disponivel")
            else:
                def _hold_cap(idx, sdir=session_dir):
                    sx2, sy2 = config.TRAY_SLOT_CENTERS[idx]
                    print(f"[hold] slot {idx} ({sx2},{sy2}) por 7s ...")
                    ht = threading.Thread(
                        target=adb_io.hold, args=(sx2, sy2, 7000), daemon=True)
                    ht.start()
                    time.sleep(1.2)
                    f2 = adb_io.capture_frame()
                    if f2 is not None:
                        ts2 = datetime.datetime.now().strftime("%H%M%S")
                        dh  = sdir / "diag"
                        cv2.imwrite(str(dh / f"hold_{ts2}.png"), f2)
                        from vision import draw_overlay as _ov, read_pieces as _rp, detect_board as _db
                        ov = _ov(f2, _db(f2), _rp(f2))
                        cv2.imwrite(str(dh / f"hold_overlay_{ts2}.png"),
                                    cv2.resize(ov, None, fx=0.5, fy=0.5))
                        print(f"[hold] {dh / f'hold_{ts2}.png'}  hold_overlay")
                    ht.join()
                threading.Thread(target=_hold_cap, args=(slot,), daemon=True).start()

        # ----------------------------------------------------------------
        # Pega frame mais recente do stream (não bloqueia)
        # ----------------------------------------------------------------
        frame = adb_io.capture_frame()

        if frame is None:
            blank = np.zeros((400, 600, 3), np.uint8)
            cv2.putText(blank, "Aguardando frame do stream...", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            cv2.imshow(_WIN, blank)
            continue

        # Porta do catalogo de grades PRIMEIRO: captura -> grade -> perfil.
        # Sem perfil validado, nenhuma jogada ADB acontece (nem solver).
        if auto_play:
            auto_play, active_profile = _board_gate(frame, active_profile)
            if not auto_play:
                print("[board] autoplay pausado pelo catalogo de grades "
                      "(A=retomar manual, Q=sair)")
                continue

        show(frame, board, pieces, seq, auto_play, move_count)

        if not auto_play:
            continue

        # Rede de seguranca: sem perfil validado, sem jogadas.
        if active_profile is None:
            print("[board] sem perfil validado — aguardando catalogacao "
                  "(nada movido)")
            continue

        # Publica a geometria validada para o mapeamento de pixels
        # (_cell_size/_cell_px/_ghost_visible). Só existe após a porta.
        global _BOARD_PROFILE
        _BOARD_PROFILE = active_profile

        # ----------------------------------------------------------------
        # FLUXO: detecta estado REAL a cada ciclo -> (re)planeja se preciso ->
        # executa 1 jogada -> VERIFICA o efeito real -> so conta se efetiva.
        # Sem loop cego: seq vazia/obsoleta sempre repleneja; NOOP repetido
        # no mesmo estado congela a execucao (observando) via STAGNATION.
        # ----------------------------------------------------------------

        # 1) Detectar estado atual do tabuleiro (geometria do perfil validado)
        board = detect_board(frame, profile=active_profile)

        # 2) Ler peças da bandeja
        pieces = read_pieces(frame)
        valid = [p for p in pieces if p is not None and p.cells]

        observed_sig = _state_sig(board, pieces)
        if observed_sig != stable_state_sig:
            stable_state_sig = observed_sig
            stable_state_count = 1
        else:
            stable_state_count += 1
        if stable_state_count < 2:
            continue

        # Estado da rodada: quantas peças válidas na bandeja
        pieces_in_tray = len(valid)

        if pieces_in_tray == 0:
            no_move_streak += 1
            if no_move_streak == 1:
                total = board.rows * board.cols
                print(f"[bot] bandeja vazia — aguardando novas peças... board {int(board.grid.sum())}/{total}")
            elif no_move_streak % 5 == 0:
                total = board.rows * board.cols
                print(f"[bot] ainda sem peças ({no_move_streak}x) — board {int(board.grid.sum())}/{total}")
            time.sleep(2.0)
            continue

        # ----------------------------------------------------------------
        # Planejamento: replaneja com o estado REAL sempre que necessario.
        # Nunca dorme em loop cego com pecas na bandeja (fim do deadlock
        # do antigo "relendo"): seq vazia ou obsoleta => REPLAN imediato.
        # ----------------------------------------------------------------
        if observing_only:
            if _state_sig(board, pieces) != prev_sig:
                observing_only = False
                prev_sig = None
                stagnant_count = 0
                print("[bot] estado mudou — retomando execucao")
            else:
                time.sleep(0.5)
                continue

        need_replan = False
        replan_why = ""
        if not seq:
            need_replan, replan_why = True, "seq vazia"
        else:
            m0 = seq[0]
            p0 = next((p for p in pieces
                       if p is not None and p.id == m0.piece_index and p.cells), None)
            if p0 is None:
                need_replan, replan_why = True, f"peca {m0.piece_index} sumiu da bandeja"

        if need_replan:
            print(f"[REPLAN] motivo: {replan_why} | board {board.rows}x{board.cols} "
                  f"fill={int(board.grid.sum())} "
                  f"pecas={[p.id for p in pieces if p is not None and p.cells]}")
            seq = best_sequence(board, pieces)
            if not seq:
                no_move_streak += 1
                if no_move_streak == 1:
                    print("[bot] sem jogadas válidas p/ estado atual — ciclo "
                          "encerrado explicitamente (observando, sem loop cego)...")
                    print(f"[board grid] {board.rows}x{board.cols}")
                    for r in range(board.rows):
                        print("  " + "".join("#" if board.grid[r, c] else "." for c in range(board.cols)))
                    for p in valid:
                        g = [['.' for _ in range(max(p.width, 1))] for _ in range(max(p.height, 1))]
                        for dr, dc in p.cells:
                            if 0 <= dr < p.height and 0 <= dc < p.width:
                                g[dr][dc] = '#'
                        print(f"  peça {p.id}: {p.width}x{p.height}")
                        for row in g:
                            print("    " + "".join(row))
                elif no_move_streak % 5 == 0:
                    total = board.rows * board.cols
                    print(f"[bot] ainda sem jogadas ({no_move_streak}x) — board {int(board.grid.sum())}/{total}")
                time.sleep(2.0)
                continue

            no_move_streak = 0
            print(f"\n[plan] {len(seq)} jogadas planejadas:")
            for i, m in enumerate(seq):
                p = next((p for p in pieces if p is not None and p.id == m.piece_index), None)
                name = f"{p.width}x{p.height}" if p else "?"
                print(f"  {i+1}. slot {m.piece_index} ({name}) -> ({m.row},{m.col})")

        # 3) Executar a PRÓXIMA jogada da sequência
        move = seq[0]
        piece = next((p for p in pieces if p is not None and p.id == move.piece_index), None)
        if piece is None:
            print(f"[bot] peça {move.piece_index} não encontrada na bandeja — pulando")
            seq.pop(0)
            time.sleep(0.3)
            continue

        # Captura frame fresco AGORA para ghost-check
        before = adb_io.capture_frame()
        if before is None:
            print("[bot] falha ao capturar frame para ghost-check, tentando mesmo assim...")
            before = frame.copy() if frame is not None else None

        done = threading.Event()
        move_success = {"ok": False}
        def _swipe(m=move, p=piece, bf=before, bd=board):
            move_success["ok"] = execute_move(m, p, before_frame=bf, board=bd)
            done.set()
        threading.Thread(target=_swipe, daemon=True).start()

        status_label = f"JOGANDO slot {move.piece_index} -> ({move.row},{move.col}) [{len(seq)} restantes]"
        while not done.wait(timeout=0.033):
            k2 = cv2.waitKey(1) & 0xFF
            if k2 in (ord('q'), 27):
                auto_play = False
                break
            # Captura frame diretamente do stream (substitui cap_q.get_nowait)
            new_frame = adb_io.capture_frame()
            if new_frame is not None:
                frame = new_frame
            show(frame, board, pieces, seq, auto_play, move_count, status_label)

        if not auto_play:
            break

        adb_ok = bool(move_success["ok"])
        print(f"[MOVE SENT] slot {move.piece_index} -> ({move.row},{move.col}) "
              f"adb_return={adb_ok}")

        if not adb_ok:
            print("[EXECUTOR_FAILURE] gesto não enviado; autoplay parado")
            break

        # 4) Estabiliza e VERIFICA o estado real (nunca confia so no retorno)
        print("[bot] aguardando estabilização do tabuleiro...")
        stable_roi = tuple(active_profile["grid_top_left"] +
                           active_profile["grid_bottom_right"])
        if not adb_io.wait_stable(timeout=3.0, poll=0.15,
                                  required_stable=2, roi=stable_roi):
            print("[bot] timeout aguardando estabilização, continuando...")

        print("[POST-MOVE VERIFY] relendo estado real...")
        verdict, after_board, after_pieces = verify_post_move(
            board, pieces, move, profile=active_profile)
        print(f"  before_board:\n{_fmt_board(board)}")
        print(f"  after_board:\n{_fmt_board(after_board)}")
        print(f"  before_pieces:\n{_fmt_pieces(pieces)}")
        print(f"  after_pieces:\n{_fmt_pieces(after_pieces)}")
        print(f"  move: slot {move.piece_index} -> ({move.row},{move.col})")
        print(f"  adb_return: {adb_ok}")
        print(f"  verification_result: {verdict}")

        if verdict != "EFFECTIVE":
            print(f"[{verdict}] primeira inconsistência; autoplay parado")
            break

        # EFFECTIVE — jogada confirmada no estado real
        print("[EFFECTIVE] jogada confirmada no estado real")
        move_count += 1
        fail_streak  = 0
        prev_sig = None
        stagnant_count = 0
        print(f"  [ok] total {move_count}")
        # Usa o tabuleiro REAL re-detectado (inclui limpezas de verdade)
        if after_board is not None:
            board = after_board

        # Remove a jogada executada da sequência
        seq.pop(0)

    adb_io.stop_scrcpy_stream()
    cv2.destroyAllWindows()
    print(f"\nTotal de jogadas: {move_count}")
    print(f"Log salvo em: {session_dir / 'log.txt'}")
    tee.close()


if __name__ == "__main__":
    main()
