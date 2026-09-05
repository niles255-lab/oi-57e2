# Blocks of Bitcoin Bot

Bot experimental de automacao para Blocks of Bitcoin (Fumb Games), usando
Python/OpenCV, ADB e um Redmi Note 8.

## Estado

As camadas de board, representacao 11x11, solver e verificacao estrita possuem
testes automatizados. O transporte de toque continuo ainda esta em
investigacao: o bot nao declara uma jogada como efetiva sem comparar o board
antes/depois.

Monkey nao faz parte do fluxo de producao.

## Estrutura

- `bot.py`: fluxo visual e autoplay legado.
- `gameplay_runner.py`: runner headless fail-closed para gameplay real.
- `model.py` e `solver.py`: estado do board e DFS.
- `vision.py`: deteccao de board, pecas e diamonds.
- `board_catalog.py` e `piece_catalog.py`: memoria visual.
- `adb_io.py`: captura nativa ADB, tracing scrcpy e transporte experimental.
- `scrcpy_transport_probe.py`: harness isolado do protocolo scrcpy 4.1.
- `scrcpy_injection_probe.py`: probe de um evento DOWN/MOVE/UP.
- `test_model_solver.py` e `test_vision_gates.py`: testes de regressao.

## Configuracao local

Configure o dispositivo por ambiente, sem salvar caminhos pessoais no Git:

```powershell
$env:BLOCKS_ADB_PATH = "C:\\path\\to\\adb.exe"
$env:BLOCKS_ADB_DEVICE_ID = "serial-do-aparelho"
$env:BLOCKS_SCRCPY_PATH = "scrcpy"
$env:BLOCKS_SCRCPY_SERVER_PATH = "C:\\path\\to\\scrcpy-server"
```

Veja `config.example.py`. Se `BLOCKS_ADB_DEVICE_ID` ficar vazio, o primeiro
dispositivo ADB em estado `device` sera selecionado.

## Testes

```powershell
python -m unittest -v test_model_solver.py test_vision_gates.py
python -u gameplay_runner.py
```

O runner para na primeira falha visual. `adb_return=True` nunca e prova de
placement.

## Transporte

O probe scrcpy registra o pacote de controle, timestamps e campos do gesto.
Sucesso exige pickup visual, board alterado e `expected_cells == actual_cells`.
