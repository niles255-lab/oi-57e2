"""Example environment configuration for local development."""

import os

ADB_PATH = os.environ.get("BLOCKS_ADB_PATH", "adb")
ADB_DEVICE_ID = os.environ.get("BLOCKS_ADB_DEVICE_ID", "")
SCRCPY_PATH = os.environ.get("BLOCKS_SCRCPY_PATH", "scrcpy")
SCRCPY_SERVER_PATH = os.environ.get("BLOCKS_SCRCPY_SERVER_PATH", "scrcpy-server")
