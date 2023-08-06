'''
Train a zero-shot GAN using CLIP-based supervision.
'''
import GPUtil
import hydra
import os
import numpy as np
import random
import time

import torch
import torch.nn.functional as F

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
from omegaconf import OmegaConf

from utils.file_utils import copytree, get_dir_img_list, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainConfig
from hydra.core.config_store import ConfigStore

SAVE_SRC = False
SAVE_DST = True

# torch.backends.cudnn.benchmark = False

def train(args: TrainConfig):

    # Set up networks, optimizers.
    random.seed(os.getpid())
    print(f"Initializing networks...")
    gpus = GPUtil.getAvailable(limit=1, maxMemory=0.2, order='random')
    device = torch.device(f"cuda:{gpus[0]}")
    net = ZSSGAN(args, device=device)

    g_optim = net.get_optim()

    # Set up output directories.
    sample_dir = "sample"
    ckpt_dir   = "checkpoint"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Training loop
    fixed_z = torch.randn(8, 512, device=device)

    progress_bar = tqdm(range(args.iter), total=args.iter)
    for i in progress_bar:
        net.train()

        # list of noises, if mixing > 0, then sample_z is [tensor(N, C), tensor(N, C)] else, sample_z is [tensor(N, c)]
        sample_z = mixing_noise(args.batch, 512, args.mixing, device)
        [sampled_src, sampled_dst], loss = net(sample_z, randomize_noise=True, truncation=1)

        net.zero_grad()
        loss.backward()
        # NOTE: clean toRGB grad
        if args.phase not in ["all", "default+"]:
            net.generator_trainable.generator.to_rgbs.zero_grad()

        g_optim.step()

        progress_bar.set_description(f"Clip loss: {loss:.3f}")

        if i % args.output_interval == 0:
            net.eval()
            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

            grid_rows = int(args.n_sample ** 0.5)

            if SAVE_SRC:
                save_images(sampled_src, sample_dir, "src", args.n_sample // grid_rows, i)

            if SAVE_DST:
                save_images(sampled_dst, sample_dir, "dst", args.n_sample // grid_rows, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

@hydra.main(config_path="config", config_name="config")
def main(args: TrainConfig):
    # save snapshot of code / args before training.
    os.makedirs("code", exist_ok=True)
    copytree(os.path.join(hydra.utils.get_original_cwd(), "criteria"), "code/criteria", )
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(),"model/ZSSGAN.py"), "code/ZSSGAN.py")

    if args.style_img_dir:
        args.target_img_list = get_dir_img_list(args.style_img_dir)
    train(args)

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainConfig)


if __name__ == "__main__":
    main()