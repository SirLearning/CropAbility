"""
配置管理模块
============
支持从 YAML/TOML/环境变量/命令行参数 多层次合并配置。
全局单例通过 get_config() 访问。
"""

from __future__ import annotations

import os
import copy
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 可选 YAML 支持
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# 可选 TOML 支持 (Python 3.11+ 内建, 否则尝试 tomli)
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
        "device_ids": [0, 1],  # 默认使用两块 GPU
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
        "chunk_size": 100_000,       # 每批读取的碱基数
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "file": None,                # None → 仅 stderr
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
    """层次化配置容器，支持点号路径访问。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = copy.deepcopy(_DEFAULT_CONFIG)
        if data:
            self._deep_merge(self._data, data)
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    # 核心访问接口
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """用点号路径取值，例如 ``cfg.get('gpu.device_ids')``。"""
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """用点号路径设值。"""
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
    # 文件加载
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """从 YAML 或 TOML 文件加载，自动根据后缀选择解析器。"""
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
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)

    def _apply_env_overrides(self) -> None:
        """将 CROPABILITY_* 环境变量覆盖到对应配置项。
        
        例如 CROPABILITY_GPU__DEVICE_IDS=0,1 → gpu.device_ids=[0,1]
        """
        prefix = "CROPABILITY_"
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            config_key = env_key[len(prefix):].lower().replace("__", ".")
            # 尝试推断类型
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
# 全局单例
# ------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[Config] = None


def get_config(path: Optional[Union[str, Path]] = None) -> Config:
    """返回全局配置单例。首次调用时可传入配置文件路径初始化。"""
    global _global_config
    with _config_lock:
        if _global_config is None:
            if path is not None:
                _global_config = Config.from_file(path)
            else:
                # 自动探测项目根目录下的 cropability.yaml / cropability.toml
                for candidate in ("cropability.yaml", "cropability.yml", "cropability.toml"):
                    p = Path(candidate)
                    if p.exists():
                        _global_config = Config.from_file(p)
                        break
                else:
                    _global_config = Config()
        elif path is not None:
            # 重新初始化
            _global_config = Config.from_file(path)
    return _global_config
