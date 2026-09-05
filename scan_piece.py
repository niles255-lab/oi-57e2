#!/usr/bin/env python3
"""
Escaneador de pecas p/ catalogo visual (somente leitura + escrita em data/).
Uso (nova CLI compativel com a antiga):
  python scan_piece.py --slot <0|1|2> [--image CAMINHO] [--save]
  python scan_piece.py --all [--image CAMINHO] [--save]
  python scan_piece.py slot <0|1|2> [--image CAMINHO] [--save]   (legado)
  python scan_piece.py all [--image CAMINHO] [--save]            (legado)
  python scan_piece.py --list | --status   (somente leitura, sem ADB)
  python scan_piece.py --watch [--image CAMINHO]   (modo continuo: "." escaneia os 3 slots)
Sem --save: apenas analisa e mostra o preview (nada e escrito no catalogo).
Com --save: mostra PEÇA DETECTADA e pergunta "Salvar no catálogo? [s/N]";
  somente se responder s/S chama save_record().
NUNCA toca no jogo: sem tap/swipe/drag/solver/autoplay.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adb_io
import piece_catalog as pc


def get_frame(image_path):
    if image_path:
        fr = cv2.imread(image_path)
        assert fr is not None, image_path
        print(f"imagem: {image_path} ({fr.shape[1]}x{fr.shape[0]})", flush=True)
        return fr
    fr = adb_io.capture_frame()
    assert fr is not None, "captura falhou"
    h, w = fr.shape[:2]
    print(f"captura ao vivo: {w}x{h}", flush=True)
    return fr


def print_drawing_safe(shape, diamonds, color):
    """Imprime desenho emoji; fallback ASCII em terminal sem Unicode."""
    try:
        print(pc.emoji_drawing(shape, diamonds, color), flush=True)
    except UnicodeEncodeError:
        print(pc.ascii_drawing(shape, diamonds), flush=True)


def show_preview(slot, a):
    print("=" * 50, flush=True)
    try:
        print(pc.preview_text(slot, a), flush=True)
    except UnicodeEncodeError:
        # terminal Windows (cp1252) nao exibe emoji: usa versao ASCII
        print(pc.preview_text_ascii(slot, a), flush=True)
    if a is None or a.get("mismatch"):
        return
    match, motivo = pc.find_match(a["shape"], a["diamond_cells"])
    if match:
        print(f"\nCATALOGO: EXISTE -> {match['kind']}/{match['id']} ({motivo})", flush=True)
    else:
        print(f"\nCATALOGO: NOVO ({motivo})", flush=True)


def show_status():
    """Comando somente-leitura: CATALOGO OK (sem ADB, sem captura)."""
    try:
        print(pc.status_text(use_emoji=True), flush=True)
    except UnicodeEncodeError:
        print(pc.status_text(use_emoji=False), flush=True)


def ask_confirm(prompt="Salvar no catálogo? [s/N] "):
    """Somente s/S confirma. Qualquer outra resposta/EOF cancela."""
    try:
        ans = input(prompt)
    except EOFError:
        return False
    return ans.strip().lower() == "s"


def parse_cli(argv):
    """Suporta --slot/--all/--list/--status/--watch e legado slot/all.

    Retorna (mode, slots, image, do_save) onde mode e
    'scan'|'status'|'watch'|None.
    """
    args = list(argv)
    if not args or "-h" in args or "--help" in args or args[0] in ("help",):
        print(__doc__, flush=True)
        return (None, [], None, False)
    if "--list" in args or "--status" in args:
        return ("status", [], None, False)
    image = None
    if "--image" in args:
        try:
            image = args[args.index("--image") + 1]
        except IndexError:
            print("ERRO: --image exige um caminho.", flush=True)
            return (None, [], None, False)
    if "--watch" in args:
        return ("watch", [], image, False)
    do_save = "--save" in args
    slots = None
    if "--all" in args:
        slots = [0, 1, 2]
    else:
        for i, tok in enumerate(args):
            if tok == "--slot" and i + 1 < len(args):
                try:
                    slots = [int(args[i + 1])]
                except ValueError:
                    slots = None
                break
            if tok.startswith("--slot="):
                try:
                    slots = [int(tok.split("=", 1)[1])]
                except ValueError:
                    slots = None
                break
        if slots is None:
            if "slot" in args:
                try:
                    slots = [int(args[args.index("slot") + 1])]
                except (IndexError, ValueError):
                    slots = None
            elif "all" in args:
                slots = [0, 1, 2]
    if slots is None or not all(s in (0, 1, 2) for s in slots):
        print(__doc__, flush=True)
        return (None, [], None, False)
    return ("scan", slots, image, do_save)


def show_detection_for_confirm(a):
    """Bloco PECA DETECTADA (somente apresentacao; reusa funcoes do catalogo)."""
    print("\nPECA DETECTADA", flush=True)
    print("", flush=True)
    try:
        print(pc.emoji_drawing(a["shape"], a["diamond_cells"], a["color"]), flush=True)
    except UnicodeEncodeError:
        print(pc.ascii_drawing(a["shape"], a["diamond_cells"]), flush=True)
    print("", flush=True)
    rows, cols = pc.dims_of(a["shape"])
    print(f"Blocos: {len(a['shape'])}", flush=True)
    print(f"Dimensao: {rows}x{cols}", flush=True)
    if a["diamond_cells"]:
        print(f"Diamantes: {len(a['diamond_cells'])} em {a['diamond_cells']}", flush=True)
    else:
        print("Diamantes: nenhum", flush=True)


def show_watch_slot(slot, a):
    """Apresenta UM slot no modo --watch. Retorna status p/ o resumo.

    Retorna (status, match|None, motivo) onde status e um de:
    'vazia' | 'divergente' | 'conhecida' | 'nova'.
    Nao salva nada aqui; o save (com confirmacao) e feito pelo chamador.
    """
    if a is None:
        print(f"\nSLOT {slot}\nVAZIA (slot vazio).", flush=True)
        return ("vazia", None, "slot vazio")
    if a.get("mismatch"):
        print(f"\nSLOT {slot}\nDIVERGENTE (espelho divergiu; nada salvo).", flush=True)
        return ("divergente", None, "mismatch")
    match, motivo = pc.find_match(a["shape"], a["diamond_cells"])
    if match:
        print(f"\nSLOT {slot}\nCONHECIDA\nID: {match['kind']}/{match['id']}", flush=True)
        return ("conhecida", match, motivo)
    print(f"\nPECA NOVA - SLOT {slot}", flush=True)
    if "configuracao de diamante nova" in motivo:
        # geometria conhecida, diamante diferente: informar claramente,
        # mas tratar como nova config (segue find_match atual).
        print(f"({motivo})", flush=True)
    print("", flush=True)
    print(f"Grade normalizada: {[list(c) for c in pc.normalize_cells(a['shape'])]}", flush=True)
    print_drawing_safe(a["shape"], a["diamond_cells"], a["color"])
    print("", flush=True)
    rows, cols = pc.dims_of(a["shape"])
    print(f"Blocos: {len(a['shape'])}", flush=True)
    print(f"Dimensao: {rows}x{cols}", flush=True)
    if a["diamond_cells"]:
        print(f"Diamantes: {len(a['diamond_cells'])} em {a['diamond_cells']}", flush=True)
        show_diamond_info(a)
    else:
        print("Diamantes: nenhum", flush=True)
    return ("nova", None, motivo)


def show_diamond_info(a):
    """Classe + aparência + conhecido/novo de cada diamante (só leitura)."""
    try:
        details = pc.describe_diamonds(a)
    except Exception:
        return
    for d in details:
        cls = d.get("class") or "desconhecida"
        try:
            known, count = pc.diamond_type_status(cls)
        except Exception:
            known, count = False, 0
        status = (f"conhecido ({count} obs.)" if known
                  else "NOVO TIPO (1a vez: exemplar sera guardado)")
        print(f"  - {d['cell']}: classe {cls} [{status}]", flush=True)
        ap = d.get("appearance") or {}
        if ap:
            print(f"    HSV {ap.get('dominant_hsv')} BGR {ap.get('dominant_bgr')} "
                  f"brilho {ap.get('brightness_v')} px={ap.get('pixel_count')}",
                  flush=True)


def watch_scan_once(frame):
    """Analisa os 3 slots a partir do MESMO frame (1 captura -> 3 slots).

    Retorna lista [(slot, a, status, match, motivo)]. Somente leitura;
    nao salva, nao toca no telefone, nao executa solver.
    """
    out = []
    for slot in (0, 1, 2):
        a = pc.analyze_slot(frame, slot)
        if a is None:
            out.append((slot, a, "vazia", None, "slot vazio"))
        elif a.get("mismatch"):
            out.append((slot, a, "divergente", None, "mismatch"))
        else:
            match, motivo = pc.find_match(a["shape"], a["diamond_cells"])
            out.append((slot, a, "conhecida" if match else "nova",
                        match, motivo))
    return out


def watch_loop(image):
    """Modo continuo de catalogo: '.' escaneia os 3 slots do frame atual.

    SOMENTE CATALOGO: sem tap/swipe/drag/solver/autoplay. Cada '.'
    captura UM frame e analisa os 3 slots desse mesmo frame. Pecas
    conhecidas nunca sao salvas; pecas novas so salvam com 's'.
    """
    print("MODO CONTINUO DE CATALOGO (somente leitura + save com confirmacao)", flush=True)
    print("Jogue as 3 pecas normalmente e digite '.' para escanear os 3 slots.", flush=True)
    print("Comandos: '.' = escanear | 'q'/'sair' = encerrar", flush=True)
    while True:
        print("", flush=True)
        print("AGUARDANDO NOVO SCAN...", flush=True)
        try:
            cmd = input('Digite "." para escanear novamente (q p/ sair): ')
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando modo continuo.", flush=True)
            return
        cmd = cmd.strip().lower()
        if cmd in ("q", "quit", "exit", "sair"):
            print("Encerrando modo continuo.", flush=True)
            return
        if cmd != ".":
            continue
        print("", flush=True)
        print("ESCANEANDO SLOTS...", flush=True)
        try:
            fr = get_frame(image)
        except (AssertionError, Exception) as e:
            print(f"Falha na captura: {e}. Tente novamente.", flush=True)
            continue
        results = watch_scan_once(fr)
        resumo = {}
        for slot, a, status, match, motivo in results:
            if status in ("vazia", "divergente"):
                show_watch_slot(slot, a)
                resumo[slot] = status
            elif status == "conhecida":
                print(f"\nSLOT {slot}\nCONHECIDA\nID: {match['kind']}/{match['id']}", flush=True)
                resumo[slot] = f"conhecida -> {match['kind']}/{match['id']}"
            else:
                st, _, mo = show_watch_slot(slot, a)
                resumo[slot] = st
                if not ask_confirm():
                    print("-> cancelado pelo usuario. NADA SALVO.", flush=True)
                    continue
                rid, d = pc.save_record(a, slot)
                kind = "diamond" if a["diamond_cells"] else "normal"
                print("CATALOGO ATUALIZADO", flush=True)
                print(f"ID: {kind}/{rid}", flush=True)
                print(f"TIPO: {kind}", flush=True)
                print(f"BLOCOS: {len(a['shape'])}", flush=True)
                if a["diamond_cells"]:
                    print(f"DIAMANTES: {len(a['diamond_cells'])} em {a['diamond_cells']}", flush=True)
                else:
                    print("DIAMANTES: nenhum", flush=True)
                print(pc.ascii_drawing(a["shape"], a["diamond_cells"]), flush=True)
        print("", flush=True)
        print("SCAN COMPLETO", flush=True)
        for slot in (0, 1, 2):
            print(f"SLOT {slot} -> {resumo.get(slot, '?')}", flush=True)


def main():
    mode, slots, image, do_save = parse_cli(sys.argv[1:])
    if mode is None:
        return
    if mode == "status":
        show_status()
        return
    if mode == "watch":
        watch_loop(image)
        return

    if image is None and do_save:
        print("Lembrete: --save re-analisa a tela atual na hora.", flush=True)
    fr = get_frame(image)

    from vision import read_pieces  # noqa  (uso somente-leitura ja feito em analyze)
    for slot in slots:
        a = pc.analyze_slot(fr, slot)
        show_preview(slot, a)
        if not do_save:
            continue
        if a is None or a.get("mismatch"):
            print("-> slot vazio/divergente: nada salvo.", flush=True)
            continue
        match, motivo = pc.find_match(a["shape"], a["diamond_cells"])
        if match:
            print(f"-> DUPLICATA: {match['kind']}/{match['id']} ja existe. NADA SALVO. ({motivo})", flush=True)
            print(f"ID: {match['kind']}/{match['id']}", flush=True)
            continue
        show_detection_for_confirm(a)
        if not ask_confirm():
            print("-> cancelado pelo usuario. NADA SALVO.", flush=True)
            continue
        rid, d = pc.save_record(a, slot)
        kind = "diamond" if a["diamond_cells"] else "normal"
        print("CATALOGO ATUALIZADO", flush=True)
        print(f"ID: {kind}/{rid}", flush=True)
        print(f"({motivo}) em {d}", flush=True)
        print("DESENHO (ver no chat; terminal sem emoji usa ASCII):", flush=True)
        print_drawing_safe(a["shape"], a["diamond_cells"], a["color"])


if __name__ == "__main__":
    main()
