#!/usr/bin/env python3
"""
Cadastro manual da geometria das grades (somente leitura + escrita em data/).
Uso:
  python scan_board.py          (captura, mostra, pergunta antes de salvar)
  python scan_board.py --list   (perfis cadastrados, sem ADB)
NUNCA altera detector, solver, autoplay ou a captura além do screencap.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adb_io
import board_catalog as bc


def show_profile(p):
    n = p["grid_size"]
    print(f"GRADE {n}x{n}", flush=True)
    print(f"cantos: {p['grid_top_left']} -> {p['grid_bottom_right']}", flush=True)
    print(f"area: {p['area_px'][0]}x{p['area_px'][1]} px", flush=True)
    print(f"pitch: {p['pitch_x']} x {p['pitch_y']} px", flush=True)
    c = p["cell_centers_xy"]
    print(f"centros: {len(c)} (1o {c[0]}, ultimo {c[-1]})", flush=True)
    print(f"confianca: cols {p['col_margin_pct']}% / linhas {p['row_margin_pct']}%", flush=True)
    print(f"frame: {p['frame_size'][0]}x{p['frame_size'][1]} em {p['timestamp']}", flush=True)
    print("", flush=True)
    print(bc.ascii_map(n), flush=True)


def show_existing(p):
    print("GRADE JA CONHECIDA", flush=True)
    print("", flush=True)
    show_profile(p)


def ask(prompt):
    try:
        return input(prompt).strip().lower() == "s"
    except EOFError:
        return False


def main():
    args = sys.argv[1:]
    if "-h" in args or "--help" in args or (args and args[0] == "help"):
        print(__doc__, flush=True)
        return
    if "--list" in args:
        profs = bc.list_profiles()
        if not profs:
            print("(nenhuma grade cadastrada)", flush=True)
            return
        for p in profs:
            print(f"- {p['shape']}: cantos {p['grid_top_left']}->{p['grid_bottom_right']} "
                  f"pitch {p['pitch_x']} ({p['timestamp']})", flush=True)
        return
    fr = adb_io.capture_frame()
    if fr is None:
        print("captura falhou", flush=True)
        return
    h, w = fr.shape[:2]
    print(f"captura: {w}x{h}", flush=True)
    p = bc.analyze_board(fr)
    if p is None:
        print("grade nao detectada com confianca (nada salvo).", flush=True)
        return
    known = bc.load_profile(p["grid_size"])
    if known is not None:
        show_existing(known)
        print("", flush=True)
        if ask("Atualizar o perfil com esta captura? [s/N] "):
            d, _ = bc.save_profile(p, fr)
            print(f"perfil {p['shape']} atualizado em {d}", flush=True)
        else:
            print("mantido o perfil existente. NADA SALVO.", flush=True)
        return
    print("NOVA GRADE", flush=True)
    print("", flush=True)
    show_profile(p)
    print("", flush=True)
    if ask("Salvar este perfil? [s/N] "):
        d, _ = bc.save_profile(p, fr)
        print(f"perfil {p['shape']} salvo em {d}", flush=True)
    else:
        print("cancelado. NADA SALVO.", flush=True)


if __name__ == "__main__":
    main()
