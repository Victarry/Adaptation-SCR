from functools import partial
import math
from typing import List

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib.pyplot import text
from PIL import Image
from torch import nn

from ZSSGAN.options.train_options import TrainConfig
from ZSSGAN.utils.text_templates import (imagenet_templates,
                                         imagenet_templates_small,
                                         part_templates)


def vjp(f, v, x, create_graph=True):
    x = x.detach().requires_grad_()
    y = f(x)
    grad_x = torch.autograd.grad(y, x, v, create_graph=create_graph)[0]
    return grad_x

def jvp(f, x, u, create_graph=True):
    x = x.detach().requires_grad_()
    v = torch.ones_like(f(x))

    g = lambda v: vjp(f, v, x, create_graph=True)
    grad_y = vjp(g, u, v, create_graph=True)
    return grad_y

def jvp_fuse(y, x, u, create_graph=False):
    v = torch.zeros_like(y).requires_grad_()
    grad_x = torch.autograd.grad(y, x, v, create_graph=True)[0]
    grad_y = torch.autograd.grad(grad_x, v, u, create_graph=create_graph)[0]
    return grad_y

def hvp_fuse(y, x, u, create_graph=False):
    grad_x = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    gradgrad_x = torch.autograd.grad(grad_x, x, u, create_graph=create_graph, retain_graph=True)[0]
    return gradgrad_x

def normalize(x: torch.Tensor):
    if x.dim() == 2:
        norm = x.norm(dim=1, keepdim=True)
    elif x.dim() == 3:
        # norm = x.norm(dim=(1, 2), keepdim=True)
        norm = x.norm(dim=(2), keepdim=True)
    return x / norm

def power_iteration(y, x, v, iterations=2):
    v = normalize(v)

    diff = y-y.detach()
    dist = torch.einsum('nd, nd->n', diff, diff).mean() / 2

    for _ in range(iterations):
        hvp = hvp_fuse(dist, x, v)
        # values = torch.einsum('nd, nd->n', v, hvp)
        v = hvp
        v = normalize(v)
    return v

def rademacher_like(x):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty_like(x)
    x.random_(0, 2)  # Creates random tensor of 0s and 1s
    x[x == 0] = -1  # Turn the 0s into -1s
    return x

def get_loss_func(func_name):
    if func_name == "mse":
        return torch.nn.MSELoss(reduction='mean')
    elif func_name == "cosine":
        return lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
    elif func_name == "nmse": # norm mse
        return lambda x, y: F.mse_loss(x.norm(dim=-1), y.norm(dim=-1), reduction='mean')
    elif func_name == "mae" or func_name == "l1":
        return torch.nn.L1Loss(reduction='mean')
    else:
        raise NotImplementedError(f"loss type of {func_name} is not supported")

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()
        self.loss_func = get_loss_func(loss_type)

    def forward(self, x, y):
        return self.loss_func(x, y)

def corres_kl_loss(src_feat, target_feat):
    N = src_feat.shape[0]
    mask = torch.diag(torch.ones(N).bool().to(src_feat.device))
    src_sim =  F.cosine_similarity(src_feat.unsqueeze(0), src_feat.unsqueeze(1), dim=-1) # (N, N)
    target_sim =  F.cosine_similarity(target_feat.unsqueeze(0), target_feat.unsqueeze(1), dim=-1)
    src_sim.masked_fill_(mask, 1e-9)
    target_sim.masked_fill_(mask, 1e-9)

    loss = F.kl_div(src_sim, target_sim, reduction="batchmean")
    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def g_path_length(img, latent, noise):
    grad, = torch.autograd.grad(
        outputs=(img* noise).sum(), inputs=latent, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    return path_lengths

class CLIPLoss(torch.nn.Module):
    def __init__(self, device, args: TrainConfig, clip_model: str, adaptive_reg_weight: nn.Parameter):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device, jit=False)
        self.model.float()
        self.args = args
        self.lerp_weight = args.lerp_weight
        self.adaptive_reg_weight = adaptive_reg_weight

        self.clip_preprocess = clip_preprocess
        self.power_iterations = args.power_iterations
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        self.patch_text_directions = None

        self.direction_loss = DirectionLoss(args.direction_loss_type)
        print("direction loss type", args.direction_loss_type)

        self.relative_loss_type = args.relative_loss_type
        print("relative loss type", self.relative_loss_type)

        self.spectral_loss_type = args.spectral_loss_type
        self.spectral_loss = DirectionLoss(args.spectral_loss_type)
        self.spectral_reg_type = args.spectral_reg_type  # "dominate" | "trace"
        print("spectral loss type", args.spectral_loss_type)

        self.spectral_weight_type = args.spectral_weight_type
        self.reg_frequency = args.reg_frequency

        self.feature_space = args.feature_space

        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.lambda_global    = args.lambda_global
        self.lambda_direction = args.lambda_direction
        self.lambda_relative = args.lambda_relative
        self.lambda_diversity = args.lambda_diversity
        self.lambda_spectral = args.lambda_spectral
        self.lambda_kl = args.lambda_kl

        self.lambda_ppl = args.lambda_ppl
        self.lambda_pplc = args.lambda_pplc
        self.mean_path_length = 0

        self.iter = 0
        self.directions = []
        self.mix_direction = None
        self.update_direction = False
        if self.update_direction:
            print("Enable update direction")
        
        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

        self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                        preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                        preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor

        self.texture_loss = torch.nn.MSELoss()

        self.templates = imagenet_templates
    
    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity
    
    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:
        template_text: List[str] = self.compose_text_with_templates(class_str, self.templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str, norm=True) -> torch.Tensor:
        source_features = self.get_text_features(source_class, norm=norm)
        target_features = self.get_text_features(target_class, norm=norm)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        if norm:
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:
        with torch.no_grad():

            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
            
    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, loss_dict: dict) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class, norm=True)

        src_encoding    = self.get_image_features(src_img, norm=True)
        target_encoding = self.get_image_features(target_img, norm=True)

        loss_dict["image_direction_norm"] = torch.norm(src_encoding - target_encoding, dim=-1).mean().item()

        edit_direction = target_encoding - src_encoding
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))
        loss = self.direction_loss(edit_direction, self.target_direction).mean()
        return loss

    def global_clip_loss(self, img: torch.Tensor, target_class) -> torch.Tensor:
        if self.target_text_features is None:
            text_features = self.target_text_features = self.get_text_features(target_class).mean(dim=0, keepdim=True)
        else:
            text_features = self.target_text_features
        image_features = self.get_image_features(img)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t()

        return (1. - logits_per_image).mean()
    
    def relative_loss(self, src_img: torch.Tensor, target_img: torch.Tensor, source_latent: torch.Tensor, target_latent: torch.Tensor, train=True) -> torch.Tensor:
        src_features = self.get_image_features(src_img, norm=True)
        target_features = self.get_image_features(target_img, norm=True)

        N = src_img.shape[0]
        row, col = torch.triu_indices(N, N, 1)
        full_src_sim = src_features.unsqueeze(1) - src_features.unsqueeze(0) # (N, 1, embed_dim) - (1, N, embed_dim) -> (N, N, embed_dim)
        full_target_sim = target_features.unsqueeze(1) - target_features.unsqueeze(0)
        src_sim = full_src_sim[row, col] # (K, embed_dim)
        target_sim = full_target_sim[row, col]

        if self.relative_loss_type == "cosine":
            loss = 1 - F.cosine_similarity(src_sim, target_sim).mean()
        elif self.relative_loss_type == "pnorm":
            loss = F.mse_loss(src_sim.norm(dim=-1), target_sim.norm(dim=-1)).mean()
        elif self.relative_loss_type == "kl":
            loss = corres_kl_loss(target_features, src_features)
        else:
            raise NotImplementedError
        return loss
    
    def ppl_loss(self, target_imgs, latents):
        loss, self.mean_path_length, path_lengths = g_path_regularize(target_imgs, latents, self.mean_path_length, decay=0.01) 
        return loss

    def pplc_loss(self, source_img, source_latent, target_img, target_latent):
        noise = torch.randn_like(source_img) / math.sqrt(
            source_img.shape[2] * source_img.shape[3]
        )
        source_ppl = g_path_length(source_img, source_latent, noise)
        target_ppl = g_path_length(target_img, target_latent, noise)
        loss = F.mse_loss(source_ppl, target_ppl)

        return loss
    
    def get_embedding_features(self, image):
        N = image.shape[0]
        if self.feature_space == "clip":
            features = self.get_image_features(image)
        elif self.feature_space == "pixel":
            features = image.reshape(N, -1)
        else:
            raise NotImplementedError(f"Feature space of {self.feature_space} not implemented")
        return F.normalize(features)
    
    def generate_direction_for_latents(self, latents, z_direction):
        directions = dict()
        directions["z"] = z_direction
        n_latent = latents["w+"].shape[1]

        directions["w"] = [[jvp_fuse(latents["w"][i], latents["z"][i], v) for v in directions["z"][i]] for i in range(len(latents["z"]))] # (n_latent, n_direction)
        
        if self.spectral_weight_type == "low":
            weight = torch.tensor([1]*6+[0]*12, dtype=torch.float32).to(self.device).reshape(1, 18, 1) 
        elif self.spectral_weight_type == "mid":
            weight = torch.tensor([0]*6 + [1]*6+[0]*6, dtype=torch.float32).to(self.device).reshape(1, 18, 1) 
        elif self.spectral_weight_type == "high":
            weight = torch.tensor([0]*12 + [1]*6, dtype=torch.float32).to(self.device).reshape(1, 18, 1) 
        elif self.spectral_weight_type == "all":
            weight = torch.tensor([1]*18, dtype=torch.float32).to(self.device).reshape(1, 18, 1) 
        elif self.spectral_weight_type == "adaptive":
            weight = self.adaptive_reg_weight
            weight = F.softmax(weight, dim=1)
        else:
            raise NotImplemented(f"spectral_weight_type {self.spectral_weight_type} not exists.")
        weight = weight[:, :n_latent]

        if len(latents["z"]) == 1:
            direction_plus = [w.unsqueeze(1).repeat(1, n_latent, 1) for w in directions["w"][0]] # (n_direction)
        else:
            direction_plus = []
            for i in range(len(directions["z"][0])):
                inject_index = latents["inject_index"]
                direction1 = directions["w"][0][i].unsqueeze(1).repeat(1, inject_index, 1)
                direction2 = directions["w"][1][i].unsqueeze(1).repeat(1, n_latent-inject_index, 1)
                direction = torch.cat([direction1, direction2], 1)
                direction_plus.append(direction)
        directions["w+"] = [v * weight for v in direction_plus]
        return directions
    
    def spectral_consistency_loss(self, src_img: torch.Tensor, target_img: torch.Tensor, source_latent: torch.Tensor,  target_latent: torch.Tensor, loss_dict=None):
        ############################ Get Target Feature Vector ##########################
        src_features = self.get_embedding_features(src_img)
        target_features = self.get_embedding_features(target_img)

        #########################  Generate Direction for different space  #############################
        if self.spectral_reg_type == "dominate":
            z_direction = torch.randn_like(source_latent["z"]) # (N, d)
            z_direction = [power_iteration(target_features, target_latent["z"], z_direction, iterations=self.power_iterations).detach()]
        elif self.spectral_reg_type == "trace":
            z_direction = [[rademacher_like(z) for _ in range(3)] for z in target_latent["z"]]  # (n_latent, n_direction)
        else:
            raise NotImplementedError(f"spectral_reg_type {self.spectral_reg_type} not implemented")
        directions = self.generate_direction_for_latents(target_latent, z_direction)
        
        ############## Calculate jvp for each direction #############################
        vhv_source_list = []
        vhv_target_list = []
        for i in range(len(z_direction[0])):
            if self.args.input_space in ["z", "w"]:
                jvp_source = jvp_fuse(src_features, source_latent[self.args.input_space][0], directions[self.args.input_space][0][i], create_graph=False) # (N, d)
                jvp_target = jvp_fuse(target_features, target_latent[self.args.input_space][0], directions[self.args.input_space][0][i], create_graph=True)
                vhv_source_list.append(torch.sum(jvp_source*jvp_source, dim=1)) # (N)
                vhv_target_list.append(torch.sum(jvp_target*jvp_target, dim=1))
            elif self.args.input_space == "w+":
                jvp_source = jvp_fuse(src_features, source_latent[self.args.input_space], directions[self.args.input_space][i], create_graph=False) # (N, d)
                jvp_target = jvp_fuse(target_features, target_latent[self.args.input_space], directions[self.args.input_space][i], create_graph=True)
                vhv_source_list.append(torch.sum(jvp_source*jvp_source, dim=1)) # (N)
                vhv_target_list.append(torch.sum(jvp_target*jvp_target, dim=1))

        trace_source = torch.mean(torch.stack(vhv_source_list, dim=0), dim=0) # (N)
        trace_target = torch.mean(torch.stack(vhv_target_list, dim=0), dim=0) # (N)
        loss_dict["trace_source"] = trace_source.mean()
        loss_dict["trace_target"] = trace_target.mean()
        if self.args.input_space == "w+":
            delta = 1
        elif self.args.input_space == 'z':
            delta = 100
        elif self.args.input_space == 'w':
            delta = 20
        loss = F.huber_loss(trace_source, trace_target, delta=delta)
        return loss
    
    def kl_loss(self, src_feat: torch.Tensor, target_feat: torch.Tensor):
        with torch.set_grad_enabled(False):
            batch_size, n_latent  = src_feat[0].shape[0], len(src_feat)
            feat_ind = np.random.randint(1, n_latent - 1, size=batch_size)

            # computing source distances
            dist_source = torch.zeros(batch_size, batch_size-1).to(self.device)

            # iterating over different elements in the batch
            for pair1 in range(batch_size):
                tmpc = 0
                # comparing the possible pairs
                for pair2 in range(batch_size):
                    if pair1 != pair2:
                        anchor_feat = torch.unsqueeze(
                            src_feat[feat_ind[pair1]][pair1].reshape(-1), 0)
                        compare_feat = torch.unsqueeze(
                            src_feat[feat_ind[pair1]][pair2].reshape(-1), 0)
                        dist_source[pair1, tmpc] = F.cosine_similarity(anchor_feat, compare_feat)
                        tmpc += 1
            dist_source = F.softmax(dist_source, dim=1)

        # computing distances among target generations
        dist_target = torch.zeros([batch_size, batch_size-1]).to(self.device)

        # iterating over different elements in the batch
        for pair1 in range(batch_size):
            tmpc = 0
            for pair2 in range(batch_size):  # comparing the possible pairs
                if pair1 != pair2:
                    anchor_feat = torch.unsqueeze(
                        target_feat[feat_ind[pair1]][pair1].reshape(-1), 0)
                    compare_feat = torch.unsqueeze(
                        target_feat[feat_ind[pair1]][pair2].reshape(-1), 0)
                    dist_target[pair1, tmpc] = F.cosine_similarity(anchor_feat, compare_feat)
                    tmpc += 1
        dist_target = F.softmax(dist_target, dim=1)
        rel_loss = F.kl_div(torch.log(dist_target), dist_source) # distance consistency loss 
        return rel_loss
    
    def diversity_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_features = self.get_image_features(src_img)
        target_features = self.get_image_features(target_img)

        N = src_img.shape[0]
        row, col = torch.triu_indices(N, N, 1)
        src_sim = src_features.unsqueeze(1) - src_features.unsqueeze(0) # (N, 1, embed_dim) - (1, N, embed_dim) -> (N, N, embed_dim)
        target_sim = target_features.unsqueeze(1) - target_features.unsqueeze(0)
        loss = torch.norm(src_sim[row, col], dim=-1) / torch.norm(target_sim[row, col], dim=-1)
        return loss.mean()
    
    def calculate_adaptive_weight(self, direction_loss, reg_loss, last_layer):
        direction_grads = torch.autograd.grad(direction_loss, last_layer, retain_graph=True)
        reg_grads = torch.autograd.grad(reg_loss, last_layer, retain_graph=True, allow_unused=True)

        direction_grads = torch.cat([x.reshape(-1) for x in direction_grads])
        reg_grads = torch.cat([x.reshape(-1) for x in reg_grads if x is not None])

        reg_weight = torch.norm(direction_grads) / (torch.norm(reg_grads) + 1e-4)
        reg_weight = torch.clamp(reg_weight, 0.0, 1e4).detach()
        return reg_weight.item()

    def forward(self, src_img: torch.Tensor, target_img: torch.Tensor, source_latent=None, target_latent=None, generator = None, train=True):
        clip_loss = 0.0
        self.iter += 1
        loss_dict = {}

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, self.args.target_class)

        if self.lambda_direction:
            direction_loss = self.clip_directional_loss(src_img, self.args.source_class, target_img, self.args.target_class, loss_dict=loss_dict)
            clip_loss += self.lambda_direction * direction_loss
            loss_dict["direction_loss"] = direction_loss.item()

        if self.lambda_relative:
            relative_loss = self.relative_loss(src_img, target_img, source_latent, target_latent, train=train)

            if self.args.adaptive_weight:
                adaptive_weight = self.calculate_adaptive_weight(direction_loss, relative_loss, last_layer=generator.get_torgbs_weight())
            else:
                adaptive_weight = 1
            clip_loss += adaptive_weight * self.lambda_relative * relative_loss
            loss_dict["relative_loss"] = relative_loss.item()
            loss_dict["relative_adaptive_weight"] = adaptive_weight
        
        if self.lambda_kl:
            kl_loss = self.kl_loss(source_latent["feat"], target_latent["feat"])
            clip_loss += self.lambda_kl * kl_loss
            loss_dict["kl_loss"] = kl_loss.item()
        
        if self.lambda_ppl:
            ppl_loss = self.ppl_loss(target_img, target_latent["w+"])
            clip_loss += self.lambda_ppl * ppl_loss

        if self.lambda_pplc:
            pplc_loss = self.pplc_loss(src_img, source_latent["w+"], target_img, target_latent["w+"])
            if self.args.adaptive_weight:
                adaptive_weight = self.calculate_adaptive_weight(direction_loss, pplc_loss, last_layer=generator.get_torgbs_weight())
            else:
                adaptive_weight = 1
            clip_loss += adaptive_weight * self.lambda_pplc * pplc_loss
            loss_dict["pplc_loss"] = pplc_loss.item()
            loss_dict["pplc_adaptive_weight"] = adaptive_weight
        
        if self.lambda_diversity:
            diversity_loss = self.diversity_loss(src_img, target_img)
            if self.args.adaptive_weight:
                adaptive_weight = self.calculate_adaptive_weight(direction_loss, diversity_loss, last_layer=generator.get_torgbs_weight())
            else:
                adaptive_weight = 1
            clip_loss += adaptive_weight * self.lambda_diversity * diversity_loss 
            loss_dict["diversity_loss"] = diversity_loss.item()
            loss_dict["diversity_adaptive_weight"] = adaptive_weight
        
        if self.lambda_spectral > 0 and self.iter % self.reg_frequency == 0:
            spectral_loss =  self.spectral_consistency_loss(src_img, target_img, source_latent, target_latent, loss_dict=loss_dict)
            if self.args.adaptive_weight:
                adaptive_weight = self.calculate_adaptive_weight(direction_loss, spectral_loss, last_layer=generator.get_torgbs_weight())
            else:
                adaptive_weight = 1

            clip_loss += adaptive_weight * self.lambda_spectral * spectral_loss
            loss_dict["spectral_loss"] = spectral_loss.item()
            loss_dict["spectral_adaptive_weigth"] = adaptive_weight

        return clip_loss, loss_dict
