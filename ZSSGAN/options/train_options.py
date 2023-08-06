from argparse import ArgumentParser
from optparse import Option
from typing import List, Optional
from utils.file_utils import get_dir_img_list
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
 

@dataclass
class TrainConfig:
    task: str = MISSING
    experiment: Optional[str] = None
    seed: int = 3
    comment: str = ""
    exp_path: str = ""

    frozen_gen_ckpt: str = ""
    train_gen_ckpt: str  = ""
    output_dir: str = MISSING
    lambda_direction: float = 1.0
    direction_loss_type: str = "cosine"

    # Non direction losses are unused in the paper. They are left here for those who want to experiment. #
    lambda_patch: float = 0.0
    lambda_global: float = 0.0
    lambda_texture: float = 0.0
    lambda_manifold: float = 0.0
    lambda_relative: float = 0.0
    lambda_diversity: float = 0.0
    lambda_contrastive: float = 0.0
    lambda_ppl: float = 0.0
    lambda_pplc: float = 0.0
    lambda_kl: float = 0.0

    # Hyperparameters about spectral loss
    lambda_spectral: float = 0.0
    input_space: str = "w+" # z | w+ | w
    feature_space: str = "clip" # clip | pixel | lpips
    spectral_loss_type: str = "nmse" # nmse | mse | l1
    spectral_reg_type: str = "trace" # dominate | trace
    spectral_weight_type: str = "adaptive" # all | mid | low | adaptive
    power_iterations: int = 3
    lerp_weight: float = 0
    adaptive_weight: bool = True # use adaptive loss weight to balance spectral loss and directional loss
    reg_frequency: int = 1

    relative_loss_type: str = 'cosine'

    save_interval: Optional[int] = 50
    output_interval: int = 100

    source_class: str = MISSING
    target_class: str = MISSING

    phase: Optional[str] = "default" # 

    sample_truncation: float = 0.7
    auto_layer_iters: int = 1
    auto_layer_k: int = 18
    auto_layer_batch: int = 8
    clip_models: List[str] = field(default_factory= lambda : ["ViT-B/32"])
    clip_model_weights: List[float] = field(default_factory= lambda:  [1.0])

    num_grid_outputs: int = 1
    crop_for_cars: bool = False

    #######################################################
    # Arguments for image style targets (instead of text) #
    #######################################################
    style_img_dir: Optional[str] = None
    img2img_batch: int = 16
    target_img_list: Optional[List[str]] = None
    #################################

    iter: int = 1000
    batch: int = 4
    n_sample: int = 8
    size: int = 1024

    mixing: float = 0.0
    ckpt: Optional[str] = None

    lr: float = 2e-3
    channel_multiplier: int = 2
    exp_name: str = MISSING
