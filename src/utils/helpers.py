import os
import random
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml


def get_logger(name: str = "HedgeFundForecasting") -> logging.Logger:
    """Cau hinh logger chuan."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int = 42) -> None:
    """Co dinh seed de tai lap ket qua thuc nghiem."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Doc file cau hinh YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_plotting_theme() -> None:
    """Thiet lap giao dien bieu do dark mode."""
    try:
        import matplotlib
        matplotlib.rcParams.update({
            "figure.dpi": 120,
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "text.color": "#c9d1d9",
            "grid.color": "#21262d",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "font.family": "monospace",
        })
    except ImportError:
        pass
