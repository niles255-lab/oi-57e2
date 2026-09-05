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

- `blocks_bot/`: codigo de producao, modelo, solver, visao e transporte.
- `tools/`: runner headless, scanners de catalogo e probes scrcpy.
- `tests/`: testes de regressao automatizados.
- `data/visual_memory/`: perfis de board, pecas e diamonds.
- `INICIAR_BOT.bat`: launcher para `python -m blocks_bot.bot`.

Diagnosticos historicos e capturas de sessoes nao fazem parte do repositorio;
eles ficam ignorados ou sao removidos para evitar que um log de uma partida
seja confundido com codigo de producao.

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
python -m unittest discover -s tests -v
python -u -m tools.gameplay_runner
```

O runner para na primeira falha visual. `adb_return=True` nunca e prova de
placement.

## Transporte

O probe scrcpy registra o pacote de controle, timestamps e campos do gesto.
Sucesso exige pickup visual, board alterado e `expected_cells == actual_cells`.
