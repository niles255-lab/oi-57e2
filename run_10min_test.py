#!/usr/bin/env python3
"""
Teste real de 10 minutos do bot (observabilidade externa, SEM alterar o bot).
- Roda bot.py como subprocesso por 600s (auto-play normal via ADB).
- Copia o stdout com timestamp p/ log_timestamped.txt na pasta da sessao.
- Em padroes de anomalia: screencap ADB (somente leitura) + contexto em diag/.
- Ao final: tenta saida graciosa (tecla 'q'); senao encerra; limpa monkey/scrcpy;
  gera resumo a partir do log da sessao.
Uso: python run_10min_test.py [segundos]
"""

import sys
import os
import re
import time
import datetime
import subprocess
import ctypes

import config

PROJECT = os.path.dirname(os.path.abspath(__file__))
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 600
ADB = config.ADB_PATH
DEVICE = config.ADB_DEVICE_ID
MAX_ANOMALIES = 25

ANOMALY_RES = [
    (re.compile(r"traceback", re.I), "traceback"),
    (re.compile(r"exception", re.I), "exception"),
    (re.compile(r"\berro\b|\berror\b|falhou|falha|FALHOU", re.I), "erro"),
    (re.compile(r"timeout", re.I), "timeout"),
    (re.compile(r"reconect|reiniciando monkey", re.I), "reconexao"),
    (re.compile(r"reject", re.I), "reject"),
    (re.compile(r"sem jogadas|sem pecas|bandeja vazia", re.I), "sem-pecas"),
    (re.compile(r"ghost nao encontrado", re.I), "ghost-nao-encontrado"),
    (re.compile(r"WinError", re.I), "winerror"),
]


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def adb(*args, timeout=25):
    try:
        return subprocess.run([ADB, "-s", DEVICE] + list(args),
                              capture_output=True, timeout=timeout)
    except Exception as e:
        print(f"[runner] adb falhou: {e}", flush=True)
        return None


def screencap_png(path):
    """Screencap ADB somente-leitura -> PNG. Retorna True/False."""
    try:
        import numpy as np
        import cv2
        res = adb("exec-out", "screencap", "-p", timeout=20)
        if res is None or res.returncode != 0 or len(res.stdout) < 1000:
            return False
        buf = np.frombuffer(res.stdout, dtype=np.uint8)
        fr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if fr is None:
            return False
        cv2.imwrite(path, fr)
        return True
    except Exception as e:
        print(f"[runner] screencap falhou: {e}", flush=True)
        return False


def send_q_to_bot_window():
    """Tenta encerrar o bot com tecla 'q' na janela OpenCV (saida graciosa)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "BlockBlaster Bot")
        if not hwnd:
            return False
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.6)
        if user32.GetForegroundWindow() != hwnd:
            return False
        user32.keybd_event(0x51, 0, 0, 0)
        time.sleep(0.1)
        user32.keybd_event(0x51, 0, 2, 0)
        return True
    except Exception as e:
        print(f"[runner] envio de 'q' falhou: {e}", flush=True)
        return False


def list_scrcpy_windows():
    """Retorna [(hwnd, titulo)] das janelas visiveis com 'scrcpy' no titulo."""
    found = []
    try:
        user32 = ctypes.windll.user32

        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value and "scrcpy" in buf.value.lower():
                    found.append((hwnd, buf.value))
            return True

        CMP = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(CMP(cb), 0)
    except Exception as e:
        print(f"[runner] EnumWindows falhou: {e}", flush=True)
    return found


def bring_to_front(title):
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        pass
    return False


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def win_rect(hwnd):
    """(x, y, w, h) da janela completa. None se invalida."""
    try:
        user32 = ctypes.windll.user32
        r = _Rect()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return None
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)
    except Exception:
        return None


def client_rect(hwnd):
    """(x, y, w, h) da area cliente em coordenadas de tela. None se invalida."""
    try:
        user32 = ctypes.windll.user32
        r = _Rect()
        if not user32.GetClientRect(hwnd, ctypes.byref(r)):
            return None
        w, h = r.right - r.left, r.bottom - r.top
        pt = _Point(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        return (pt.x, pt.y, w, h)
    except Exception:
        return None


def overlap_area(a, b):
    """Area de intersecao entre retangulos (x, y, w, h)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def move_window(hwnd, x, y):
    try:
        ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, 0, 0,
                                          0x0001 | 0x0004 | 0x0010)
        return True
    except Exception:
        return False


def find_window(title, timeout=25.0):
    user32 = ctypes.windll.user32
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                return hwnd
        except Exception:
            pass
        time.sleep(0.3)
    return 0


def device_present():
    try:
        r = subprocess.run([ADB, "devices"], capture_output=True, text=True, timeout=10)
        return DEVICE in r.stdout
    except Exception:
        return False


def parse_summary(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    boards, oks, grids = [], [], []
    pecas_det = 0
    rescued = 0
    errs = {"traceback": 0, "erro": 0, "timeout": 0, "reconexao": 0,
            "reject": 0, "sem-pecas": 0, "ghost": 0, "exec-fail": 0}
    for ln in lines:
        m = re.search(r"\[board\] (\d+)/\d+ cells", ln)
        if m:
            boards.append(int(m.group(1)))
        m = re.search(r"\[ok\] total (\d+)", ln)
        if m:
            oks.append(int(m.group(1)))
        m = re.search(r"\[grid\] detectado (\d+)x(\d+)", ln)
        if m:
            grids.append((int(m.group(1)), int(m.group(2))))
        m = re.search(r"PECA \d+: \S+, blocos=(\d+)", ln)
        if m and int(m.group(1)) >= 2:
            pecas_det += 1
        if "celula(s) com diamante em" in ln:
            rescued += 1
        low = ln.lower()
        if "traceback" in low:
            errs["traceback"] += 1
        if "falhou" in low or " winerror" in low or "erro" in low:
            errs["erro"] += 1
        if "timeout" in low:
            errs["timeout"] += 1
        if "reconect" in low or "reiniciando monkey" in low:
            errs["reconexao"] += 1
        if "reject" in low:
            errs["reject"] += 1
        if "sem jogadas" in low or "sem pecas" in low or "bandeja vazia" in low:
            errs["sem-pecas"] += 1
        if "ghost nao encontrado" in low:
            errs["ghost"] += 1
        if "execute_move falhou" in low:
            errs["exec-fail"] += 1
    grid_changes = sum(1 for a, b in zip(grids, grids[1:]) if a != b)
    max_run = 1
    run = 1
    for a, b in zip(boards, boards[1:]):
        if a == b:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return {"lines": len(lines), "cycles": len(boards), "moves": max(oks) if oks else 0,
            "boards": boards, "grids": grids, "grid_changes": grid_changes,
            "errs": errs, "pecas_det": pecas_det, "rescued": rescued,
            "distinct_fills": len(set(boards)), "max_same_run": max_run}


def main():
    # --- pre-voo (sem tocar no jogo) ---
    if not device_present():
        print(f"[runner] ABORTADO: dispositivo {DEVICE} ausente no adb", flush=True)
        return
    wins = list_scrcpy_windows()
    print(f"[runner] janelas scrcpy visiveis: {wins if wins else 'nenhuma'}", flush=True)
        foreign = [t for _, t in wins if t != f"scrcpy_bot_{DEVICE}"]
    if foreign:
        print(f"[runner] ABORTADO: scrcpy estranho aberto {foreign} — "
              f"feche antes (conflito de stream)", flush=True)
        return
    try:
        user32 = ctypes.windll.user32
        if user32.FindWindowW(None, "BlockBlaster Bot"):
            print("[runner] ABORTADO: bot ja em execucao", flush=True)
            return
    except Exception:
        pass

    t0 = time.time()
    print(f"[runner] iniciando bot.py por {DURATION}s (sem alterar codigo)", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", "bot.py"],
        cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace")

    session_dir = None
    ts_log = None
    recent = []
    last_grid = None
    anomalies = 0
    last_anom = {}
    gate_ok = False
    gate_deadline = None
    brought_front = False
    finished_early = False

    def fire(reason, ctx):
        nonlocal anomalies
        if session_dir is None or anomalies >= MAX_ANOMALIES:
            return
        now = time.time()
        if now - last_anom.get(reason, 0) < 10:
            return
        last_anom[reason] = now
        anomalies += 1
        tag = datetime.datetime.now().strftime("%H%M%S")
        base = os.path.join(session_dir, "diag", f"anomaly_{tag}_{reason}")
        ok = screencap_png(base + ".png")
        try:
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(f"motivo: {reason}\nlinha: {ctx}\n\n--- contexto ---\n")
                f.write("\n".join(recent[-40:]))
        except Exception:
            pass
        print(f"[runner] anomalia '{reason}' -> {os.path.basename(base)}.png (ok={ok})",
              flush=True)

    gate_ok = False
    gate_deadline = None
    deadline = None
    abort_cause = None
    positioned = False
    scrcpy_ref = None
    pending_erro = False
    erro_streak = 0
    zero_streak = 0
    max_erro_streak = 0
    max_zero_streak = 0
    last_watch = 0.0
    user32 = ctypes.windll.user32

    def place_bot_window():
        """Move a janela do bot para a borda direita (0% overlap c/ scrcpy)."""
        try:
            hb = user32.FindWindowW(None, "BlockBlaster Bot")
            if not hb:
                return None
            r = win_rect(hb)
            if not r:
                return None
            sw = user32.GetSystemMetrics(0)
            nx = sw - r[2] - 8
            user32.SetWindowPos(hb, 0, nx, 0, 0, 0,
                                0x0001 | 0x0004 | 0x0010)
            time.sleep(0.4)
            return win_rect(hb)
        except Exception:
            return None

    while True:
        if deadline is not None and time.time() > deadline:
            print("[runner] tempo completo; solicitando saida graciosa (q)...", flush=True)
            break
        if time.time() - t0 > DURATION + 300:
            abort_cause = "timeout absoluto (gate nunca passou)"
            print(f"[runner] ABORT: {abort_cause}", flush=True)
            break
        if abort_cause is not None:
            print(f"[runner] ABORT: {abort_cause}", flush=True)
            break
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            print("[runner] bot encerrou sozinho antes do tempo", flush=True)
            break
        if line == "":
            time.sleep(0.05)
            continue
        if session_dir is None:
            m = re.search(r"Session:\s*(\S+)", line)
            if m:
                session_dir = m.group(1).strip()
                ts_log = open(os.path.join(session_dir, "log_timestamped.txt"),
                              "w", encoding="utf-8", buffering=1)
                print(f"[runner] sessao: {session_dir}", flush=True)
                gate_deadline = time.time() + 60
        if ts_log:
            ts_log.write(f"[{ts()}] {line}")
        recent.append(line.rstrip("\n"))
        recent = recent[-60:]

        # posicionamento 0% overlap (uma vez, apos as duas janelas existirem)
        if not positioned and session_dir is not None:
            hs = user32.FindWindowW(None, f"scrcpy_bot_{DEVICE}")
            hb = user32.FindWindowW(None, "BlockBlaster Bot")
            if hs and hb:
                sc = client_rect(hs)
                if sc is not None:
                    scrcpy_ref = sc
                    place_bot_window()
                    rb = win_rect(hb)
                    ov = overlap_area(tuple(sc), tuple(rb)) if rb else -1
                    print(f"[runner] scrcpy cliente={sc} bot={rb} overlap={ov}px "
                          f"({100 * ov / (sc[2] * sc[3]):.1f}%)", flush=True)
                    if ov == 0:
                        positioned = True
                        print("[runner] overlap 0% confirmado", flush=True)
                    else:
                        abort_cause = f"overlap {ov}px apos posicionar"
                        continue

        # watchdog a cada 5s: scrcpy parado + overlap ainda 0%
        if positioned and time.time() - last_watch > 5:
            last_watch = time.time()
            hs = user32.FindWindowW(None, f"scrcpy_bot_{DEVICE}")
            if not hs:
                abort_cause = "janela scrcpy sumiu"
                continue
            cur = client_rect(hs)
            if cur is None or abs(cur[2] - scrcpy_ref[2]) > 4 or abs(cur[3] - scrcpy_ref[3]) > 4:
                abort_cause = f"scrcpy moveu/redimensionou {scrcpy_ref} -> {cur}"
                continue
            hb = user32.FindWindowW(None, "BlockBlaster Bot")
            if hb:
                rb = win_rect(hb)
                if rb and overlap_area(tuple(cur), tuple(rb)) > 0:
                    place_bot_window()
                    rb2 = win_rect(hb)
                    ov2 = overlap_area(tuple(cur), tuple(rb2)) if rb2 else -1
                    fire("overlap-reposition",
                         f"bot reposicionado; overlap agora {ov2}px")
                    if ov2 != 0:
                        abort_cause = "overlap persistente apos reposicionar"
                        continue

        # GATE: primeiro ciclo com tabuleiro real (preenchido + peca + sem ERRO)
        if gate_ok is not True:
            m = re.search(r"\[board\] (\d+)/\d+ cells", line)
            if m and int(m.group(1)) > 5:
                gate_ok = "aguardando-peca"
            if gate_ok and re.search(r"PECA \d+: \S+, blocos=(\d+)", line):
                gate_ok = True
                deadline = time.time() + DURATION
                print(f"[runner] GATE OK: captura mostra jogo real "
                      f"({time.time() - t0:.0f}s apos inicio); contando "
                      f"{DURATION}s a partir de agora", flush=True)
            if "ERRO de deteccao" in line:
                gate_ok = False
            if gate_deadline and time.time() > gate_deadline and gate_ok is not True:
                abort_cause = "gate falhou: sem tabuleiro real em 60s"
                continue

        # raias de invalidez pos-gate (param o teste em vez de loopar no erro)
        if "ERRO de deteccao" in line:
            pending_erro = True
        m = re.search(r"\[board\] (\d+)/\d+ cells", line)
        if m and gate_ok is True:
            if pending_erro:
                erro_streak += 1
                max_erro_streak = max(max_erro_streak, erro_streak)
            else:
                erro_streak = 0
            pending_erro = False
            if int(m.group(1)) == 0:
                zero_streak += 1
                max_zero_streak = max(max_zero_streak, zero_streak)
            else:
                zero_streak = 0
            if erro_streak >= 8:
                abort_cause = f"{erro_streak} ciclos seguidos com ERRO de grade"
                continue
            if zero_streak >= 6:
                abort_cause = f"{zero_streak} boards 0/64 seguidos (captura perdeu o jogo)"
                continue

        m = re.search(r"\[grid\] detectado (\d+)x(\d+)", line)
        if m:
            g = (int(m.group(1)), int(m.group(2)))
            if last_grid is not None and g != last_grid:
                fire("grid-change",
                     f"grade {last_grid[0]}x{last_grid[1]} -> {g[0]}x{g[1]}")
            last_grid = g
            continue
        for rx, reason in ANOMALY_RES:
            if rx.search(line):
                fire(reason, line.strip()[:160])
                break

    # --- encerramento ---
    if proc.poll() is None:
        if send_q_to_bot_window():
            try:
                proc.wait(timeout=15)
                print("[runner] bot saiu graciosamente", flush=True)
            except subprocess.TimeoutExpired:
                print("[runner] 'q' sem efeito; encerrando processo", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        else:
            print("[runner] janela nao encontrada; encerrando processo", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if ts_log:
        ts_log.close()

    # --- limpeza: monkey no aparelho + scrcpy local (somente desta sessao) ---
    adb("shell", "pkill", "-f", "monkey", timeout=10)
    try:
        subprocess.run(["taskkill", "/F", "/IM", "scrcpy.exe"],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    adb("forward", "--remove", "tcp:1080", timeout=10)

    # --- resumo ---
    if session_dir:
        log_path = os.path.join(session_dir, "log.txt")
        s = parse_summary(log_path) if os.path.exists(log_path) else None
        try:
            diag_files = sorted(os.listdir(os.path.join(session_dir, "diag")))
        except Exception:
            diag_files = []
        rep = [f"SESSAO: {session_dir}",
               f"gate captura real: {'OK' if gate_ok is True else 'FALHOU/NAO-ATINGIDO'}",
               f"posicionamento 0% overlap: {'OK' if positioned else 'FALHOU/NAO-ATINGIDO'}",
               f"scrcpy ref: {scrcpy_ref}",
               f"causa de parada antecipada: {abort_cause or 'nenhuma (tempo completo ou saida do bot)'}",
               f"maior raia ERRO-grade pos-gate: {max_erro_streak}",
               f"maior raia board-0 pos-gate: {max_zero_streak}",
               f"ciclos(detecoes [board]): {s['cycles'] if s else '?'}",
               f"jogadas([ok] max): {s['moves'] if s else '?'}",
               f"pecas detectadas (linhas PECA c/ >=2 blocos): {s['pecas_det'] if s else '?'}",
               f"diamantes resgatados (pecas finais): {s['rescued'] if s else '?'}",
               f"trocas de grade: {s['grid_changes'] if s else '?'} "
               f"(grades vistas: {sorted(set(s['grids'])) if s else '?'})",
               f"fills distintos: {s['distinct_fills'] if s else '?'} | "
               f"maior seq. fills iguais: {s['max_same_run'] if s else '?'}",
               f"erros: {s['errs'] if s else '?'}",
               f"anomalias capturadas pelo runner: {anomalies}",
               f"arquivos em diag/ ({len(diag_files)}):",
               *[f'  {f}' for f in diag_files],
               "nota: dx/dy por peca nao constam no log do bot "
               "(so pitch/score; dx/dy exigiriam alterar o bot)"]
        # heuristica: board inalterado apos jogada
        if s:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
            fills, oks_i = [], []
            for i, ln in enumerate(lines):
                m = re.search(r"\[board\] (\d+)/\d+ cells", ln)
                if m:
                    fills.append((i, int(m.group(1))))
                m = re.search(r"\[ok\] total (\d+)", ln)
                if m:
                    oks_i.append((i, int(m.group(1))))
            same = 0
            for li, n in oks_i:
                pre = [f for (i, f) in fills if i < li]
                post = [f for (i, f) in fills if i > li]
                if pre and post and pre[-1] == post[0]:
                    same += 1
            rep.append(f"jogadas com board aparentemente inalterado (heuristica): {same}")
        summary = "\n".join(rep)
        print("=" * 60 + "\nRESUMO FINAL\n" + "=" * 60 + "\n" + summary, flush=True)
        try:
            with open(os.path.join(session_dir, "resumo_10min.txt"),
                      "w", encoding="utf-8") as f:
                f.write(summary + "\n")
        except Exception:
            pass
    print("[runner] fim.", flush=True)


if __name__ == "__main__":
    main()
