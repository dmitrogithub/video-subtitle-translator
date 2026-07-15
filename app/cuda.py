"""NVIDIA runtime discovery and diagnostics for faster-whisper."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


_DLL_HANDLES: list[object] = []
_REGISTERED_DIRECTORIES: set[str] = set()


def discover_nvidia_dll_directories(search_paths: list[str] | None = None) -> list[Path]:
    """Find CUDA DLL folders installed by NVIDIA Python wheels on Windows."""
    if os.name != "nt":
        return []
    directories: list[Path] = []
    for root_value in search_paths if search_paths is not None else sys.path:
        if not root_value:
            continue
        root = Path(root_value)
        for component in ("cublas", "cudnn", "cuda_runtime"):
            candidate = root / "nvidia" / component / "bin"
            if candidate.is_dir() and candidate not in directories:
                directories.append(candidate)
    return directories


def prepare_nvidia_cuda_dll_directories() -> list[Path]:
    """Make CUDA 12 DLLs from installed Python wheels available to CTranslate2."""
    directories = discover_nvidia_dll_directories()
    if not directories:
        return []
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized_entries = {os.path.normcase(item) for item in path_entries if item}
    additions = [
        str(directory)
        for directory in directories
        if os.path.normcase(str(directory)) not in normalized_entries
    ]
    if additions:
        os.environ["PATH"] = os.pathsep.join([*additions, *path_entries])
    dll_adder = getattr(os, "add_dll_directory", None)
    if dll_adder is not None:
        for directory in directories:
            normalized = os.path.normcase(str(directory))
            if normalized not in _REGISTERED_DIRECTORIES:
                _DLL_HANDLES.append(dll_adder(str(directory)))
                _REGISTERED_DIRECTORIES.add(normalized)
    return directories


def ctranslate2_cuda_report() -> dict[str, Any]:
    """Return non-throwing CUDA diagnostics without loading a Whisper model."""
    directories = prepare_nvidia_cuda_dll_directories()
    report: dict[str, Any] = {
        "dll_directories": [str(item) for item in directories],
        "ctranslate2_installed": False,
        "cuda_device_count": 0,
        "cuda_available": False,
        "error": None,
    }
    try:
        import ctranslate2  # type: ignore[import-not-found]

        report["ctranslate2_installed"] = True
        report["cuda_device_count"] = int(ctranslate2.get_cuda_device_count())
        report["cuda_available"] = report["cuda_device_count"] > 0
    except Exception as error:
        report["error"] = str(error)
    return report
