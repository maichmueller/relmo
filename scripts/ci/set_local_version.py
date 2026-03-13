from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_FILE = Path(__file__).resolve().parents[2] / "pyproject.toml"
VERSION_RE = re.compile(r'(?m)^(version = ")([^"]+)(")$')


def main() -> int:
    local = os.getenv("RELM_WHEEL_LOCAL_VERSION", "").strip()
    if not local:
        print("RELM_WHEEL_LOCAL_VERSION not set; leaving pyproject version unchanged.")
        return 0

    text = PROJECT_FILE.read_text()
    match = VERSION_RE.search(text)
    if match is None:
        raise SystemExit(f"Could not locate project version in {PROJECT_FILE}.")

    base = match.group(2).split("+", 1)[0]
    version = f"{base}+{local}"
    updated = VERSION_RE.sub(rf'\g<1>{version}\g<3>', text, count=1)
    PROJECT_FILE.write_text(updated)
    print(f"Set local wheel version to {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
