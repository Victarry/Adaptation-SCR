import torch
import math
import random

def mixing_noise(batch, latent_dim, prob, device, truncation=False):
    if prob > 0 and random.random() < prob:
            noises = torch.randn(2, batch, latent_dim, device=device).unbind(0) # [Tensor, Tensor]
    else:
            noises = [torch.randn(batch, latent_dim, device=device)] # [Tensor]
    return noises
