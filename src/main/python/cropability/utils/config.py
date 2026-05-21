"""
Configuration management
========================
Supports layered configuration from YAML/TOML, environment variables, and CLI arguments.
Access the global singleton via get_config().
"""

from __future__ import annotations

import os
import copy
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional YAML support
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Optional TOML support (built-in in Python 3.11+, otherwise try tomli)
try:
    import tomllib as _tomllib
    _TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
        _TOML_AVAILABLE = True
    except ImportError:
        _TOML_AVAILABLE = False


_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "CropAbility",
        "version": "0.1.0",
        "description": "Plant genomics HPC framework",
    },
    "gpu": {
        "enabled": True,
        "device_ids": [0, 1],  # Default: use two GPUs
        "memory_fraction": 0.90,
        "allow_growth": True,
        "backend": "cuda",           # cuda | cpu
        "mixed_precision": True,     # bf16 on H100, fp16 on A2
    },
    "compute": {
        "batch_size": 128,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 2,
        "block_size": 1024,          # Triton kernel block size default
    },
    "genomics": {
        "reference_genome": None,
        "min_base_quality": 20,
        "min_mapping_quality": 30,
        "max_read_length": 300,
        "kmer_size": 31,
    },
    "io": {
        "input_dir": "./data/input",
        "output_dir": "./data/output",
        "tmp_dir": "./data/tmp",
        "chunk_size": 100_000,       # Bases read per batch
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "file": None,                # None → stderr only
        "rotate": True,
        "max_bytes": 50 * 1024 * 1024,
        "backup_count": 5,
    },
    "distributed": {
        "enabled": False,
        "backend": "nccl",
        "master_addr": "localhost",
        "master_port": 29500,
        "world_size": 1,
        "rank": 0,
    },
}


class Config:
    """Hierarchical configuration container with dot-path access."""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = copy.deepcopy(_DEFAULT_CONFIG)
        if data:
            self._deep_merge(self._data, data)
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    # Core access API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot path, e.g. ``cfg.get('gpu.device_ids')``."""
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """Set a value by dot path."""
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load from a YAML or TOML file; parser is chosen from the suffix."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML config. pip install pyyaml")
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif suffix == ".toml":
            if not _TOML_AVAILABLE:
                raise ImportError("tomllib/tomli is required for TOML config.")
            with path.open("rb") as f:
                data = _tomllib.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        return cls(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    def _apply_env_overrides(self) -> None:
        """Apply CROPABILITY_* environment variables to matching config keys.

        Example: CROPABILITY_GPU__DEVICE_IDS=0,1 → gpu.device_ids=[0,1]
        """
        prefix = "CROPABILITY_"
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            config_key = env_key[len(prefix):].lower().replace("__", ".")
            # Infer type when possible
            parsed: Any = env_val
            if env_val.lower() in ("true", "false"):
                parsed = env_val.lower() == "true"
            elif "," in env_val:
                try:
                    parsed = [int(x.strip()) for x in env_val.split(",")]
                except ValueError:
                    parsed = [x.strip() for x in env_val.split(",")]
            else:
                try:
                    parsed = int(env_val)
                except ValueError:
                    try:
                        parsed = float(env_val)
                    except ValueError:
                        parsed = env_val
            self.set(config_key, parsed)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[Config] = None


def get_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Return the global config singleton. Pass a config file path on first call to initialize."""
    global _global_config
    with _config_lock:
        if _global_config is None:
            if path is not None:
                _global_config = Config.from_file(path)
            else:
                # Auto-detect cropability.yaml / cropability.toml in project root
                for candidate in ("cropability.yaml", "cropability.yml", "cropability.toml"):
                    p = Path(candidate)
                    if p.exists():
                        _global_config = Config.from_file(p)
                        break
                else:
                    _global_config = Config()
        elif path is not None:
            # Re-initialize
            _global_config = Config.from_file(path)
    return _global_config
