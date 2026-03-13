from __future__ import annotations

import os
import platform
import subprocess
import sys


INDEX_BY_CHANNEL = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    version = os.getenv("RELM_TORCH_VERSION", "").strip()
    channel = os.getenv("RELM_TORCH_CHANNEL", "").strip().lower()
    if not version or not channel:
        print("RELM_TORCH_VERSION/RELM_TORCH_CHANNEL not set; skipping torch install.")
        return 0
    if channel not in INDEX_BY_CHANNEL:
        raise SystemExit(f"Unsupported RELM_TORCH_CHANNEL={channel!r}.")
    if platform.system() == "Darwin" and channel != "cpu":
        raise SystemExit("macOS native wheels only support the CPU torch lane.")

    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    cmd = [sys.executable, "-m", "pip", "install", f"torch=={version}"]
    if not (platform.system() == "Darwin" and channel == "cpu"):
        cmd.extend(
            [
                "--index-url",
                INDEX_BY_CHANNEL[channel],
                "--extra-index-url",
                "https://pypi.org/simple",
            ]
        )
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
