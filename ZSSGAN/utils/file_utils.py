import os
import shutil
from typing import List, Tuple
from pathlib import Path
import torch
from torchvision import utils

import cv2

def get_dir_img_list(dir_path, valid_exts=[".png", ".jpg", ".jpeg"]):
    dir_path = Path(dir_path)
    if dir_path.is_dir():
        file_list = [file_name for file_name in dir_path.iterdir() if file_name.suffix in valid_exts]
    else:
        file_list = [dir_path]
    return file_list

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def save_images(images: torch.Tensor, output_dir: str, file_prefix: str, nrows: int, iteration: int) -> None:
    utils.save_image(
        images,
        os.path.join(output_dir, f"{file_prefix}_{str(iteration).zfill(6)}.jpg"),
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )

def save_torch_img(img: torch.Tensor, output_dir: str, file_name: str) -> None:
    img = img.permute(1, 2, 0).cpu().detach().numpy()

    img = img[:, :, ::-1] # RGB to BGR for cv2 saving
    cv2.imwrite(os.path.join(output_dir, file_name), img)

def resize_img(img: torch.Tensor, size: Tuple[int]) -> torch.Tensor:
    if not isinstance(size, Tuple):
        size = (size, size)
    return torch.nn.functional.interpolate(img.unsqueeze(0), size)[0]

def save_paper_image_grid(sampled_images: torch.Tensor, sample_dir: str, file_name: str):
    img = (sampled_images + 1.0) * 126 # de-normalize

    half_size = (img.size()[-2] // 2, img.size()[-1] // 2) 
    quarter_size = (half_size[0] // 2, half_size[1] // 2)

    base_fig = torch.cat([img[0], img[1]], dim=2)
    sub_cols = [torch.cat([resize_img(img[i + j], half_size) for j in range(2)], dim=1) for i in range(2, 8, 2)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    sub_cols = [torch.cat([resize_img(img[i + j], quarter_size) for j in range(4)], dim=1) for i in range(8, 16, 4)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    save_torch_img(base_fig, sample_dir, file_name)

def save_paper_animal_grid(sampled_images: torch.Tensor, sample_dir: str, file_name: str):

    img = (sampled_images + 1.0) * 126 # de-normalize

    half_size = img.size()[-1] // 2
    quarter_size = half_size // 2

    base_fig = torch.cat([img[0]], dim=2)
    sub_cols = [torch.cat([resize_img(img[i + j], half_size) for j in range(2)], dim=1) for i in range(1, 5, 2)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    sub_cols = [torch.cat([resize_img(img[i + j], quarter_size) for j in range(4)], dim=1) for i in range(5, 13, 4)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    save_torch_img(base_fig, sample_dir, file_name)