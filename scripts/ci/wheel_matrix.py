from __future__ import annotations

import argparse
import json


BUILD_TORCH = "2.8.0"
COMPAT_TORCH_VERSIONS = ("2.8.0", "2.9.0", "2.10.0")
RELEASE_PYTHONS = "cp310-* cp311-* cp312-* cp313-*"
PR_PYTHONS = "cp311-*"

CPU_RELEASE = [
    {
        "name": "linux-x86_64-cpu",
        "runner": "ubuntu-24.04",
        "torch_version": BUILD_TORCH,
        "torch_channel": "cpu",
        "local_version": "cpu",
        "find_links_page": "relmo-cpu.html",
        "cibw_archs": "x86_64",
        "cibw_build": RELEASE_PYTHONS,
    },
    {
        "name": "windows-x86_64-cpu",
        "runner": "windows-2022",
        "torch_version": BUILD_TORCH,
        "torch_channel": "cpu",
        "local_version": "cpu",
        "find_links_page": "relmo-cpu.html",
        "cibw_archs": "AMD64",
        "cibw_build": RELEASE_PYTHONS,
    },
    {
        "name": "macos-arm64-cpu",
        "runner": "macos-14",
        "torch_version": BUILD_TORCH,
        "torch_channel": "cpu",
        "local_version": "cpu",
        "find_links_page": "relmo-cpu.html",
        "cibw_archs": "arm64",
        "cibw_build": RELEASE_PYTHONS,
    },
]

CUDA_RELEASE = [
    {
        "name": "linux-x86_64-cu126",
        "runner": "ubuntu-24.04",
        "torch_version": BUILD_TORCH,
        "torch_channel": "cu126",
        "local_version": "cu126",
        "find_links_page": "relmo-cu126.html",
        "manylinux_image": "sameli/manylinux_2_34_x86_64_cuda_12.6",
        "cibw_archs": "x86_64",
        "cibw_build": RELEASE_PYTHONS,
    },
    {
        "name": "linux-x86_64-cu128",
        "runner": "ubuntu-24.04",
        "torch_version": BUILD_TORCH,
        "torch_channel": "cu128",
        "local_version": "cu128",
        "find_links_page": "relmo-cu128.html",
        "manylinux_image": "sameli/manylinux_2_34_x86_64_cuda_12.8",
        "cibw_archs": "x86_64",
        "cibw_build": RELEASE_PYTHONS,
    },
]


def pr_cpu() -> list[dict[str, str]]:
    lane = dict(CPU_RELEASE[0])
    lane["cibw_build"] = PR_PYTHONS
    return [lane]


GROUPS = {
    "pr-cpu": pr_cpu,
    "release-cpu": lambda: CPU_RELEASE,
    "release-cuda": lambda: CUDA_RELEASE,
    "release-cpu-pages": lambda: [
        {
            "name": "cpu",
            "find_links_page": "relmo-cpu.html",
            "local_version": "cpu",
        }
    ],
    "release-cuda-pages": lambda: [
        {
            "name": "cu126",
            "find_links_page": "relmo-cu126.html",
            "local_version": "cu126",
        },
        {
            "name": "cu128",
            "find_links_page": "relmo-cu128.html",
            "local_version": "cu128",
        }
    ],
    "pr-cpu-compat": lambda: [
        {
            "name": f"linux-cpu-torch{version}",
            "artifact_name": "wheelhouse-linux-x86_64-cpu",
            "runner": "ubuntu-24.04",
            "python_version": "3.11",
            "torch_version": version,
            "torch_channel": "cpu",
        }
        for version in COMPAT_TORCH_VERSIONS
    ],
    "release-cpu-compat": lambda: [
        {
            "name": f"linux-cpu-torch{version}",
            "artifact_name": "wheelhouse-linux-x86_64-cpu",
            "runner": "ubuntu-24.04",
            "python_version": "3.11",
            "torch_version": version,
            "torch_channel": "cpu",
        }
        for version in COMPAT_TORCH_VERSIONS
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    args = parser.parse_args()
    print(json.dumps(GROUPS[args.group](), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
