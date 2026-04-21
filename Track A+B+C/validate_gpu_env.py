#!/usr/bin/env python3
"""
Validate exact Python package versions and core GPU runtime assumptions.
"""

import argparse
import importlib
from importlib import metadata
from pathlib import Path
import sys


def parse_requirements(path):
    expected = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        expected[name.strip()] = version.strip()
    return expected


def import_name(package_name):
    mapping = {
        "Pillow": "PIL",
        "pycocotools": "pycocotools",
    }
    return mapping.get(package_name, package_name.replace("-", "_"))


def assert_importable(package_name):
    module_name = import_name(package_name)
    importlib.import_module(module_name)


def main():
    parser = argparse.ArgumentParser(description="Validate GPU Python environment for this project.")
    parser.add_argument("--requirements", default="Deformable-DETR/requirements.txt")
    parser.add_argument("--expect-cuda", action="store_true", help="Fail if CUDA is not available in torch")
    parser.add_argument("--project-root", default="Deformable-DETR", help="Project root used for op import checks")
    args = parser.parse_args()

    requirements_path = Path(args.requirements).resolve()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))

    expected = parse_requirements(requirements_path)
    print(f"Validating requirements from {requirements_path}")

    failures = []

    for package_name, expected_version in expected.items():
        try:
            installed_version = metadata.version(package_name)
            if installed_version != expected_version:
                failures.append(
                    f"{package_name}: expected {expected_version}, found {installed_version}"
                )
            assert_importable(package_name)
            print(f"[OK] {package_name}=={installed_version}")
        except Exception as exc:
            failures.append(f"{package_name}: {exc}")

    try:
        import torch
        import torchvision

        print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"[INFO] torch.version.cuda = {torch.version.cuda}")
        print(f"[INFO] torchvision version = {torchvision.__version__}")
        if args.expect_cuda and not torch.cuda.is_available():
            failures.append("CUDA is not available in torch, but --expect-cuda was requested")
    except Exception as exc:
        failures.append(f"torch runtime check failed: {exc}")

    try:
        from models.ops.modules.ms_deform_attn import MSDeformAttn  # noqa: F401

        print("[OK] Imported MSDeformAttn module")
    except Exception as exc:
        failures.append(f"MSDeformAttn import failed: {exc}")

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("\nEnvironment validation passed.")


if __name__ == "__main__":
    main()
