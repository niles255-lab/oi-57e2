import subprocess
import socket
import time
import io
import threading
import numpy as np
import cv2
from typing import Optional, Tuple
import config
import ctypes
import mss

# ---------------------------------------------------------------------------
# Sistema de coordenadas: tudo no bot/visao opera em pixels NATIVOS do
# aparelho (Redmi Note 8: 1080x2340). Qualquer frame capturado em outra
# escala (ex.: janela scrcpy via MSS) e normalizado para NATIVE em
# capture_frame(), unico ponto de conversao do pipeline.
# ---------------------------------------------------------------------------
NATIVE_W: int = 1080
NATIVE_H: int = 2340

# ---------------------------------------------------------------------------
# Monkey driver (touch preciso: down / move / up independentes)
# ---------------------------------------------------------------------------

_monkey_sock: Optional[socket.socket] = None
_monkey_proc = None


def monkey_start(port: int = 1080) -> bool:
    global _monkey_sock, _monkey_proc
    try:
        _cmd(["shell", "pkill", "-f", "monkey"], timeout=3.0)
        time.sleep(0.5)
        _monkey_proc = subprocess.Popen(
            [config.ADB_PATH, "-s", config.ADB_DEVICE_ID,
             "shell", "monkey", "--port", str(port), "-v"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        subprocess.run(
            [config.ADB_PATH, "-s", config.ADB_DEVICE_ID,
             "forward", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True, timeout=5,
        )
        _monkey_sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        _monkey_sock.settimeout(4.0)
        print(f"[monkey] conectado na porta {port}")
        return True
    except Exception as e:
        _monkey_sock = None
        if config.DEBUG:
            print(f"[monkey] start error: {e}")
        return False


def monkey_stop():
    global _monkey_sock, _monkey_proc
    try:
        if _monkey_sock:
            _monkey_sock.close()
        if _monkey_proc:
            _monkey_proc.terminate()
    except Exception:
        pass
    _monkey_sock = None
    _monkey_proc = None


def _monkey_cmd(cmd: str) -> bool:
    global _monkey_sock
    if _monkey_sock is None:
        return False
    try:
        _monkey_sock.sendall((cmd.strip() + "\n").encode())
        resp = _monkey_sock.recv(256).decode().strip()
        return resp == "OK"
    except Exception as e:
        if config.DEBUG:
            print(f"[monkey] cmd error '{cmd}': {e}")
        try:
            _monkey_sock.close()
        except Exception:
            pass
        _monkey_sock = None
        return False


def monkey_reconnect(port: int = 1080, retries: int = 2) -> bool:
    """Tenta reiniciar o monkey se a conexao caiu."""
    global _monkey_sock, _monkey_proc
    print("[monkey] reconectando...")
    try:
        fw = subprocess.run(
            [config.ADB_PATH, "forward", "--list"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"[monkey] diagnostico forward: "
              f"{fw.stdout.strip() or '(vazio)'}")
    except Exception as e:
        print(f"[monkey] diagnostico forward falhou: {e}")
    try:
        ps = subprocess.run(
            [config.ADB_PATH, "-s", config.ADB_DEVICE_ID, "shell",
             "ps -A | grep -i monkey"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"[monkey] diagnostico ps: "
              f"{ps.stdout.strip() or '(sem processo monkey no aparelho)'}")
    except Exception as e:
        print(f"[monkey] diagnostico ps falhou: {e}")
    for _ in range(retries):
        if monkey_start(port):
            return True
        time.sleep(1.0)
    print("[monkey] reconexao falhou — usando swipe como fallback")
    return False


def touch_down(x: int, y: int) -> bool:
    return _monkey_cmd(f"touch down {x} {y}")


def touch_move(x: int, y: int) -> bool:
    return _monkey_cmd(f"touch move {x} {y}")


def touch_up(x: int, y: int) -> bool:
    return _monkey_cmd(f"touch up {x} {y}")


def monkey_ready() -> bool:
    return _monkey_sock is not None


def _cmd(args, timeout: float = 8.0):
    cmd = [config.ADB_PATH]
    if config.ADB_DEVICE_ID:
        cmd += ["-s", config.ADB_DEVICE_ID]
    cmd += args
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def check_device() -> bool:
    try:
        res = subprocess.run(
            [config.ADB_PATH, "devices"],
            capture_output=True, text=True, timeout=5,
        )
        devices = [
            line.split("\t", 1)[0]
            for line in res.stdout.splitlines()
            if "\tdevice" in line
        ]
        if config.ADB_DEVICE_ID:
            return config.ADB_DEVICE_ID in devices
        if devices:
            config.ADB_DEVICE_ID = devices[0]
            return True
        return False
    except Exception:
        return False


def _capture_adb_frame() -> Optional[np.ndarray]:
    for attempt in range(2):
        try:
            res = _cmd(["exec-out", "screencap", "-p"], timeout=15.0)
            if res.returncode != 0 or len(res.stdout) < 1000:
                time.sleep(0.3)
                continue
            buf = np.frombuffer(res.stdout, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception as e:
            if config.DEBUG:
                print(f"[ADB] capture error (tentativa {attempt+1}): {e}")
            time.sleep(0.5)
    return None


def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = None) -> bool:
    if duration_ms is None:
        duration_ms = config.DRAG_DURATION_MS
    try:
        hold_ms = getattr(config, 'DRAG_PRE_HOLD_MS', 0)
        if hold_ms > 0:
            # Batched continuous events are the only ADB path observed to move
            # this Unity surface. Verification remains authoritative because
            # the shell process may outlive the injected UP event.
            steps = max(1, int(round(float(duration_ms) / 100.0)))
            interval_s = float(duration_ms) / 1000.0 / steps
            proc = subprocess.Popen(
                [config.ADB_PATH, "-s", config.ADB_DEVICE_ID, "shell"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                def send_event(kind, x, y):
                    proc.stdin.write(
                        f"input touchscreen motionevent {kind} "
                        f"{int(x)} {int(y)}\n".encode())
                    proc.stdin.flush()

                send_event("DOWN", x1, y1)
                time.sleep(float(hold_ms) / 1000.0)
                for i in range(1, steps + 1):
                    time.sleep(interval_s)
                    xi = int(round(x1 + (x2 - x1) * i / steps))
                    yi = int(round(y1 + (y2 - y1) * i / steps))
                    send_event("MOVE", xi, yi)
                send_event("UP", x2, y2)
                proc.stdin.write(b"exit\n")
                proc.stdin.flush()
                proc.stdin.close()
                try:
                    proc.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                return True
            except Exception:
                return False
        res = _cmd(
            ["shell", "input", "touchscreen", "swipe",
             str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            timeout=float(duration_ms) / 1000.0 + 5.0,
        )
        return res.returncode == 0
    except Exception as e:
        if config.DEBUG:
            print(f"[ADB] swipe error: {e}")
        return False


def hold(x: int, y: int, duration_ms: int = 7000) -> bool:
    """Segura o dedo em (x, y) por duration_ms ms sem mover."""
    try:
        res = _cmd(
            ["shell", "input", "touchscreen", "swipe",
             str(x), str(y), str(x), str(y), str(duration_ms)],
            timeout=float(duration_ms) / 1000.0 + 5.0,
        )
        return res.returncode == 0
    except Exception as e:
        if config.DEBUG:
            print(f"[ADB] hold error: {e}")
        return False


def tap(x: int, y: int) -> bool:
    try:
        res = _cmd(["shell", "input", "tap", str(x), str(y)], timeout=5.0)
        return res.returncode == 0
    except Exception:
        return False


def frames_differ(f1: np.ndarray, f2: np.ndarray, roi: Tuple = None, threshold: int = 500) -> bool:
    if roi is not None:
        x1, y1, x2, y2 = roi
        f1 = f1[y1:y2, x1:x2]
        f2 = f2[y1:y2, x1:x2]
    diff = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return int(np.sum(gray > 30)) > threshold


def wait_stable(timeout: float = 4.0, poll: float = 0.15,
                required_stable: int = 2, roi: Tuple = None) -> bool:
    board_roi = roi or (
        config.GRID_TOP_LEFT[0], config.GRID_TOP_LEFT[1],
        config.GRID_BOTTOM_RIGHT[0], config.GRID_BOTTOM_RIGHT[1],
    )
    last = capture_frame()
    if last is None:
        time.sleep(timeout)
        return False
    stable = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        cur = capture_frame()
        if cur is None:
            continue
        if not frames_differ(last, cur, roi=board_roi):
            stable += 1
            if stable >= required_stable:
                return True
        else:
            stable = 0
        last = cur
    return False


# =============================================================================
# Scrcpy Stream Capture (novo - substitui screencap individual)
# =============================================================================

class _ScrcpyStream:
    """Gerencia stream contínuo do scrcpy via MSS."""
    
    def __init__(self, device_id: str, window_title: str, max_size: str = "1080"):
        self.device_id = device_id
        self.window_title = window_title
        self.max_size = max_size
        
        # Tenta encontrar scrcpy.exe
        self.scrcpy_path = self._find_scrcpy()
        if not self.scrcpy_path:
            raise RuntimeError("scrcpy.exe não encontrado. Instale via 'winget install Genymobile.scrcpy'")
        
        self._proc = None
        self._hwnd = 0
        self._monitor = None
        self._sct = None
        self._last_frame = None
        self._frame_counter = 0
        self._last_frame_ts = 0.0
        self._running = False
        self._thread = None
    
    def _find_scrcpy(self) -> Optional[str]:
        """Procura scrcpy.exe em locais conhecidos."""
        import shutil
        
        # 1. Tenta PATH do sistema
        path = shutil.which("scrcpy")
        if path:
            return path
        
        # 2. Tenta local padrão do winget (padrão por usuário)
        import os
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            winget_path = os.path.join(
                local_appdata, "Microsoft", "WinGet", "Packages",
                "Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe",
                "scrcpy-win64-v4.1", "scrcpy.exe"
            )
            if os.path.exists(winget_path):
                return winget_path
        
        # 3. Tenta local padrão do scoop
        scoop_path = os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "apps", "scrcpy", "current", "scrcpy.exe")
        if os.path.exists(scoop_path):
            return scoop_path
        
        return None
    
    def _find_window(self, timeout: float = 15.0) -> bool:
        user32 = ctypes.windll.user32
        start = time.time()
        while time.time() - start < timeout:
            hwnd = user32.FindWindowW(None, self.window_title)
            if hwnd:
                self._hwnd = hwnd
                return True
            time.sleep(0.1)
        return False
    
    def _get_client_rect(self) -> Tuple[int, int, int, int]:
        """Retorna (left, top, width, height) da área cliente em coordenadas de tela."""
        user32 = ctypes.windll.user32
        
        class Rect(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        class Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        
        client_rect = Rect()
        if not user32.GetClientRect(self._hwnd, ctypes.byref(client_rect)):
            return 0, 0, 0, 0
        cw = client_rect.right - client_rect.left
        ch = client_rect.bottom - client_rect.top
        
        pt = Point(0, 0)
        if not user32.ClientToScreen(self._hwnd, ctypes.byref(pt)):
            return 0, 0, 0, 0
        
        return pt.x, pt.y, cw, ch
    
    def _wait_ready(self, min_width: int = 300, min_height: int = 500, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            left, top, width, height = self._get_client_rect()
            if width >= min_width and height >= min_height:
                return True
            time.sleep(0.2)
        return False
    
    def start(self) -> bool:
        """Inicia scrcpy e prepara captura MSS."""
        try:
            # Inicia scrcpy
            cmd = [
                self.scrcpy_path,
                f"--window-title={self.window_title}",
                f"--max-size={self.max_size}",
                f"--serial={self.device_id}",
            ]
            if config.DEBUG:
                print(f"[scrcpy_stream] Iniciando: {' '.join(cmd)}")
            
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # Aguarda janela
            if not self._find_window():
                if config.DEBUG:
                    print("[scrcpy_stream] Janela não encontrada")
                return False
            
            # Aguarda área cliente pronta
            if not self._wait_ready():
                if config.DEBUG:
                    print("[scrcpy_stream] Área cliente não ficou pronta")
                return False
            
            # Configura monitor MSS
            left, top, width, height = self._get_client_rect()
            self._monitor = {"left": left, "top": top, "width": width, "height": height}
            self._sct = mss.mss()
            
            if config.DEBUG:
                print(f"[scrcpy_stream] Capturando área cliente: {width}x{height} em ({left},{top})")
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            
            return True
            
        except Exception as e:
            if config.DEBUG:
                print(f"[scrcpy_stream] Erro ao iniciar: {e}")
            return False
    
    def _capture_loop(self):
        """Loop contínuo de captura - mantém apenas o frame mais recente."""
        while self._running:
            try:
                img = self._sct.grab(self._monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self._last_frame = frame
                self._frame_counter += 1
                self._last_frame_ts = time.monotonic()
            except Exception:
                pass
            time.sleep(0.001)  # ~1000 fps max, limitado pelo MSS
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Retorna o frame mais recente (cópia)."""
        if self._last_frame is not None:
            return self._last_frame.copy()
        return None

    def get_latest_frame_info(self):
        """Return (frame, counter, age_seconds) for diagnostic consumers."""
        frame = self.get_latest_frame()
        if frame is None:
            return None, self._frame_counter, None
        age = max(0.0, time.monotonic() - self._last_frame_ts)
        return frame, self._frame_counter, age
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sct:
            self._sct.close()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# Instância global do stream
_scrcpy_stream: Optional[_ScrcpyStream] = None


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Converte qualquer frame capturado para a resolucao NATIVA (1080x2340).

    Unico ponto de conversao de escala do pipeline: depois daqui, visao,
    overlay, ghost-check e controle falam sempre a mesma coordenada fisica
    do aparelho, independente da escala da janela scrcpy.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w == NATIVE_W and h == NATIVE_H:
        return frame
    if config.DEBUG:
        print(f"[capture] normalizando frame {w}x{h} -> {NATIVE_W}x{NATIVE_H}")
    return cv2.resize(frame, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)


def capture_stream_frame() -> Optional[np.ndarray]:
    """Return the latest scrcpy frame, normalized for intermediate tracing."""
    if _scrcpy_stream is None:
        return None
    frame = _scrcpy_stream.get_latest_frame()
    return normalize_frame(frame) if frame is not None else None


def capture_frame() -> Optional[np.ndarray]:
    """
    Captura frame SEMPRE em resolucao NATIVA (1080x2340).

    Captura ADB nativa primeiro; scrcpy fica reservado para tracing visual.
    Retorna numpy.ndarray BGR em coordenadas nativas, ou None.
    """
    frame = _capture_adb_frame()
    if frame is not None:
        return normalize_frame(frame)
    return capture_stream_frame()


def init_scrcpy_stream(device_id: str = None, window_title: str = None, max_size: str = "1080") -> bool:
    """Inicializa o stream scrcpy. Deve ser chamado uma vez no início do bot."""
    global _scrcpy_stream
    if _scrcpy_stream is not None:
        return True
    
    device_id = device_id or config.ADB_DEVICE_ID
    if not device_id and not check_device():
        return False
    device_id = device_id or config.ADB_DEVICE_ID
    window_title = window_title or f"scrcpy_bot_{device_id}"
    
    _scrcpy_stream = _ScrcpyStream(device_id, window_title, max_size)
    return _scrcpy_stream.start()


def stop_scrcpy_stream():
    """Para o stream scrcpy."""
    global _scrcpy_stream
    if _scrcpy_stream is not None:
        _scrcpy_stream.stop()
        _scrcpy_stream = None
