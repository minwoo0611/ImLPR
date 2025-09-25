# Warsaw University of Technology

import os
import time
import configparser
from typing import Optional
import numpy as np


def get_datetime() -> str:
    return time.strftime("%Y%m%d_%H%M")


class ModelParams:
    def __init__(self, model_params_path: str):
        cfg = configparser.ConfigParser()
        cfg.read(model_params_path)

        p = cfg["BACKBONE"]
        self.model_name = p.get("model_name", "dinov2_vits14")
        self.num_trainable_blocks = p.getint("num_trainable_blocks", 2)
        self.adapter_frequency = p.getint("adapter_frequency", 3)

        p = cfg["AGGREGATOR"]
        self.num_channels = p.getint("num_channels", 384)
        self.num_clusters = p.getint("num_clusters", 64)
        self.cluster_dim = p.getint("cluster_dim", 128)
        self.token_dim = p.getint("token_dim", 256)

    def print(self) -> None:
        print("Model parameters:")
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("")


class TrainingParams:
    """Parameters for model training (keeps original variable names)."""

    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        assert os.path.exists(params_path), f"Cannot find configuration file: {params_path}"
        assert os.path.exists(model_params_path), f"Cannot find model-specific configuration file: {model_params_path}"
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        cfg = configparser.ConfigParser()
        cfg.read(self.params_path)

        # DEFAULT
        d = cfg["DEFAULT"]
        self.model_name = d.get("model_name", "ImLPR_training")
        self.dataset_folder = d.get("dataset_folder")
        self.load_weights = d.get("load_weights", None)

        # TRAIN
        t = cfg["TRAIN"]
        self.save_freq = t.getint("save_freq", 0)
        self.num_workers = t.getint("num_workers", 0)
        self.batch_size = t.getint("batch_size", 64)

        # Optional ints/floats can be absent → None
        self.batch_split_size: Optional[int] = self._get_optional_int(t, "batch_split_size", None)
        self.batch_expansion_th: Optional[float] = self._get_optional_float(t, "batch_expansion_th", None)

        if self.batch_expansion_th is not None:
            assert 0.0 < self.batch_expansion_th < 1.0, "batch_expansion_th must be in (0,1)"
            self.batch_size_limit = t.getint("batch_size_limit", 256)
            self.batch_expansion_rate = t.getfloat("batch_expansion_rate", 1.5)
            assert self.batch_expansion_rate > 1.0, "batch_expansion_rate must be > 1"
        else:
            # keep variable names; align defaults
            self.batch_size_limit = t.getint("batch_size_limit", self.batch_size)
            self.batch_expansion_rate = None

        self.val_batch_size = t.getint("val_batch_size", self.batch_size_limit)

        self.lr = t.getfloat("lr", 1e-3)
        self.epochs = t.getint("epochs", 20)
        self.optimizer = t.get("optimizer", "Adam")
        self.scheduler = t.get("scheduler", "CustomCosineWarmupScheduler")
        self.warmup_epochs = t.getint("warmup_epochs", 10)


        self.loss = t.get("loss").lower()
        self.positives_per_query = t.getint("positives_per_query", 4)
        self.tau1 = t.getfloat("tau1", 0.01)
        self.margin = self._get_optional_float(t, "margin", None)

        self.num_positive_pairs = t.getint("num_positive_pairs", 216)
        self.num_pos = t.getint("num_pos", 192)
        self.num_hn_samples = t.getint("num_hn_samples", 64)
        self.temperature = t.getfloat("temperature", 0.20)
        self.vdist = t.getint("vdist", 3)
        self.hdist = t.getint("hdist", 20)
        self.circular_horizontal = t.getboolean("circular_horizontal", True)

        # Image / sensor
        self.image_H = t.getint("image_H", 1022)
        self.image_W = t.getint("image_W", 126)
        self.uFoV = t.getfloat("uFoV", 11.5)
        self.dFoV = t.getfloat("dFoV", -10.1)
        self.sensor_range = t.getfloat("sensor_range", 200.0)

        # Similarity
        self.similarity = t.get("similarity", "euclidean")
        assert self.similarity in ["cosine", "euclidean"], "similarity must be 'cosine' or 'euclidean'"

        # Data files 
        self.train_file = t.get("train_file")
        self.val_file = t.get("val_file", None)
        self.test_file = t.get("test_file", None)

        # Model-specific
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()

    @staticmethod
    def _get_optional_int(section: configparser.SectionProxy, key: str, default: Optional[int]) -> Optional[int]:
        return section.getint(key) if section.get(key, fallback="") != "" else default

    @staticmethod
    def _get_optional_float(section: configparser.SectionProxy, key: str, default: Optional[float]) -> Optional[float]:
        return section.getfloat(key) if section.get(key, fallback="") != "" else default

    def _check_params(self) -> None:
        assert os.path.exists(self.dataset_folder), f"Cannot access dataset: {self.dataset_folder}"

    def print(self) -> None:
        print("Parameters:")
        for k, v in vars(self).items():
            if k != "model_params":
                print(f"{k}: {v}")
        self.model_params.print()
