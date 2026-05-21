#!/usr/bin/env python3
"""One-shot CropAbility development environment setup.

Creates the Conda base env (if needed), installs Python deps via pip, and builds
the Rust PyO3 extension with maturin.

Examples:
    python install.py              # GPU (CUDA PyTorch + Triton)
    python install.py --cpu        # CPU-only PyTorch
    python install.py --skip-conda # pip + maturin only (env already active)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV = "cropability"
ENV_FILE = REPO_ROOT / "environment.yml"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    display = " ".join(cmd)
    print(f"+ {display}")
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env, check=True)


def _conda_exe() -> str:
    exe = shutil.which("conda")
    if not exe:
        sys.exit("conda not found on PATH. Install Miniconda/Mambaforge first.")
    return exe


def _conda_env_exists(name: str) -> bool:
    import json

    result = subprocess.run(
        [_conda_exe(), "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return name in {Path(p).name for p in data.get("envs", [])}


def _ensure_conda_env(name: str) -> None:
    conda = _conda_exe()
    if _conda_env_exists(name):
        print(f"Conda env '{name}' exists; updating from {ENV_FILE.name}")
        _run([conda, "env", "update", "-n", name, "-f", str(ENV_FILE), "--prune"])
    else:
        print(f"Creating Conda env '{name}' from {ENV_FILE.name}")
        _run([conda, "env", "create", "-n", name, "-f", str(ENV_FILE)])


def _reexec_in_conda_env(name: str, argv: list[str]) -> None:
    """Re-run this script inside the target Conda env."""
    conda = _conda_exe()
    cmd = [conda, "run", "--no-capture-output", "-n", name, sys.executable, str(REPO_ROOT / "install.py")]
    cmd.extend(argv)
    _run(cmd)


def _active_env_name() -> str | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    return Path(prefix).name


def _requirements_file(cpu: bool) -> Path:
    return REPO_ROOT / ("requirements-cpu.txt" if cpu else "requirements-gpu.txt")


def _find_libclang_path(conda_prefix: Path) -> str:
    # Prefer system LLVM for bindgen; conda libclang can produce incomplete HTSlib bindings.
    system_candidates = sorted(
        Path("/usr/lib").glob("llvm-*/lib"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    candidates = [*system_candidates, conda_prefix / "lib"]
    for directory in candidates:
        if not directory.is_dir():
            continue
        if (directory / "libclang.so").exists() or any(directory.glob("libclang.so*")):
            return str(directory)
    return str(conda_prefix / "lib")


def _rust_build_env(conda_prefix: Path) -> dict[str, str]:
    env = os.environ.copy()
    bindir = conda_prefix / "bin"
    gcc = bindir / "x86_64-conda-linux-gnu-gcc"
    gxx = bindir / "x86_64-conda-linux-gnu-g++"
    if gcc.exists():
        env["CC"] = str(gcc)
    if gxx.exists():
        env["CXX"] = str(gxx)
    env["LIBCLANG_PATH"] = _find_libclang_path(conda_prefix)
    env["HTSLIB_DIR"] = str(conda_prefix)
    pkg = conda_prefix / "lib" / "pkgconfig"
    if pkg.is_dir():
        existing = env.get("PKG_CONFIG_PATH", "")
        env["PKG_CONFIG_PATH"] = f"{pkg}:{existing}" if existing else str(pkg)
    return env


def _install_pip(cpu: bool) -> None:
    req = _requirements_file(cpu)
    if not req.is_file():
        sys.exit(f"Missing {req.name}")
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def _build_rust_extension() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        sys.exit("CONDA_PREFIX is not set; activate the Conda env before building Rust.")
    maturin = shutil.which("maturin")
    if not maturin:
        sys.exit("maturin not found; run pip install step first.")
    env = _rust_build_env(Path(conda_prefix))
    _run(
        [maturin, "develop", "--release", "--features", "python,htslib"],
        env=env,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up CropAbility dev environment (Conda + pip + maturin).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU-only PyTorch (no CUDA / no Triton extra). Default is GPU.",
    )
    parser.add_argument(
        "--env-name",
        default=DEFAULT_ENV,
        help=f"Conda environment name (default: {DEFAULT_ENV}).",
    )
    parser.add_argument("--skip-conda", action="store_true", help="Skip Conda env create/update.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip pip install.")
    parser.add_argument("--skip-rust", action="store_true", help="Skip maturin develop.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = "cpu" if args.cpu else "gpu"
    print(f"CropAbility setup ({mode})")

    # Phase 1: ensure Conda env exists, then re-exec inside it unless already there.
    if not args.skip_conda:
        if _active_env_name() != args.env_name:
            _ensure_conda_env(args.env_name)
            forward = []
            if args.cpu:
                forward.append("--cpu")
            forward.extend(["--skip-conda", "--env-name", args.env_name])
            if args.skip_pip:
                forward.append("--skip-pip")
            if args.skip_rust:
                forward.append("--skip-rust")
            _reexec_in_conda_env(args.env_name, forward)
            return

    if not args.skip_pip:
        print("Installing Python dependencies…")
        _install_pip(cpu=args.cpu)

    if not args.skip_rust:
        print("Building Rust native extension…")
        _build_rust_extension()

    print()
    print("Done.")
    print(f"  conda activate {args.env_name}")
    print("  pytest src/test/python")
    print("  pytest src/test/python -m native   # needs _core")


if __name__ == "__main__":
    main()
