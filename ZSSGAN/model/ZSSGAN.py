import itertools
import sys
import os
from typing import Dict, List

from options.train_options import TrainConfig
sys.path.insert(0, os.path.abspath('../'))


import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ZSSGAN.model.sg2_model import Generator, Discriminator
from ZSSGAN.criteria.clip_loss import CLIPLoss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
        super().__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])
            #  + list(self.get_all_layers()[6][6:])
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        if phase == 'default':
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])
        if phase == 'default+':
            # everything except mapping
            all_layers = self.get_all_layers()
            return list(all_layers)[1:3] + list(self.get_all_layers()[4][:]) + list(all_layers[6])
        raise NotImplementedError(f"phase {phase} not supported")

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers
    
    def get_torgbs_weight(self):
        return [layer.conv.weight for layer in self.generator.to_rgbs]

    def forward(self,
        styles,
        return_latents=False,
        truncation=1,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True,
        return_feats=False):
        return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code, return_feats=return_feats)

class ZSSGAN(torch.nn.Module):
    def __init__(self, args: TrainConfig, device='cuda:0'):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = device
        self.logger = SummaryWriter()
        self.global_step = 0


        self.adaptive_weight = None
        if self.args.spectral_weight_type == "adaptive" and self.args.input_space == "w+":
            self.adaptive_weight = torch.nn.Parameter(torch.ones(1, 18, 1, device=self.device))

        # Set up frozen (source) generator
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size, device=self.device, channel_multiplier=args.channel_multiplier)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size, device=self.device, channel_multiplier=args.channel_multiplier)
        self.generator_trainable.freeze_layers()
        # NOTE: Add torgb into requires grad for adaptive weight calculation, and later manually clean grad
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase) + list(self.generator_trainable.get_all_layers()[6]))
        self.generator_trainable.train()

        # Losses
        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}
        self.clip_loss_models: Dict(str, CLIPLoss) = {model_name: CLIPLoss(self.device, args=args, clip_model=model_name, adaptive_reg_weight=self.adaptive_weight)
                                for model_name in args.clip_models if self.clip_model_weights[model_name] > 0}

        self.mse_loss  = torch.nn.MSELoss()

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        if args.target_img_list is not None:
            self.set_img2img_direction()
        
    def get_optim(self):
        if self.args.spectral_weight_type == "adaptive":
            assert self.args.input_space == "w+", "adaptive spectral reg must be used in w+ space."
            print("using adaptive spectral reg")
            return torch.optim.Adam([
                        {"params": self.generator_trainable.parameters()},
                        {"params": self.adaptive_weight, "lr": 1e-2, "betas": (0, 0.99)}], 
                    lr=self.args.lr, betas=(0, 0.99))
        else:
            return torch.optim.Adam(self.generator_trainable.parameters(), lr=self.args.lr, betas=(0, 0.99))

    def set_img2img_direction(self):
        with torch.no_grad():
            sample_z  = torch.randn(self.args.img2img_batch, 512, device=self.device)
            generated = self.generator_trainable([sample_z])[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)

                model.target_direction = direction

    def determine_opt_layers(self):

        sample_z = torch.randn(self.args.auto_layer_batch, 512, device=self.device)

        initial_w_codes = self.generator_frozen.style([sample_z])
        initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.n_latent, 1)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            w_codes_for_gen = w_codes.unsqueeze(0)
            generated_from_w = self.generator_trainable(w_codes_for_gen, input_is_latent=True)[0]

            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.args.target_class) for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))

            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()

        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())

        conv_layers = list(all_layers[4])
        rgb_layers = list(all_layers[6]) # currently not optimized

        idx_to_layer = all_layers[2:4] + conv_layers # add initial convs to optimization

        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx]

        # uncomment to add RGB layers to optimization.
        # for idx in chosen_layer_idx:
        #     if idx % 2 == 1 and idx >= 3 and idx < 14:
        #         chosen_layers.append(rgb_layers[(idx - 3) // 2])
        chosen_layers += rgb_layers

        # uncomment to add learned constant to optimization
        # chosen_layers.append(all_layers[1])

        return chosen_layers
    
    def generate_images_with_latent(self, latent_list: List[torch.Tensor], truncation: float, randomize_noise: bool):
        """
        z: List of latent codes, if apply style mixing, more than one latent code is in z.
        """
        source_latent = dict()
        target_latent = dict()

        source_latent["z"] = [z.detach().clone().requires_grad_(True) for z in latent_list]
        source_latent["w"] = [w.requires_grad_(True) for w in self.generator_frozen.style(source_latent["z"])] 
        if self.args.lambda_kl:
            source_image, source_latent["w+"], source_latent["feat"] = self.generator_frozen.forward(source_latent["w"], 
                input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise, return_feats=True)
        else:
            source_image, source_latent["w+"], source_latent["inject_index"] = self.generator_frozen.forward(source_latent["w"], 
                input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise, return_latents=True)

        target_latent["z"] = [z.detach().clone().requires_grad_(True) for z in latent_list]
        target_latent["w"] = [w.requires_grad_(True) for w in self.generator_trainable.style(target_latent["z"])] 
        if self.args.lambda_kl:
            target_image, target_latent["w+"], target_latent["feat"] = self.generator_trainable.forward(target_latent["w"], 
                input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise, return_feats=True)
        else:
            target_image, target_latent["w+"], target_latent["inject_index"] = self.generator_trainable.forward(target_latent["w"], 
                input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise, return_latents=True)
        
        return (source_image, target_image), (source_latent, target_latent)

    def forward(
        self,
        styles, # List of z code, tensor with shape [N, z_dim]
        truncation=1,
        randomize_noise=True,
    ):
        if self.training and self.auto_layer_iters > 0:
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)
        
        (source_image, target_image), (source_latent, target_latent) = self.generate_images_with_latent(styles, truncation, randomize_noise)

        if self.training:
            clip_losses = []
            for model_name in self.clip_model_weights.keys():
                if self.clip_model_weights[model_name] > 0:
                    loss, loss_dict = self.clip_loss_models[model_name](source_image, target_image, source_latent=source_latent, target_latent=target_latent, generator=self.generator_trainable)
                    for key, item in loss_dict.items():
                        self.logger.add_scalar(f"{model_name}/{key}", item, global_step=self.global_step)
                    clip_losses.append(self.clip_model_weights[model_name] * loss)
            clip_loss = torch.sum(torch.stack(clip_losses))
        else:
            # clip_losses = []
        #     for model_name in self.clip_model_weights.keys():
        #         if self.clip_model_weights[model_name] > 0:
        #             loss, loss_dict = self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class, source_latent=source_latent, target_latent=target_latent, train=False)
        #             for key, item in loss_dict.items():
        #                 self.logger.add_scalar(f"eval_{model_name}/{key}", item, global_step=self.global_step)
        #             clip_losses.append(self.clip_model_weights[model_name] * loss)
        #     clip_loss = torch.sum(torch.stack(clip_losses))
            clip_loss = 0

        self.global_step += 1
        if self.global_step % 50 == 0:
            self.logger.add_image("image/source", make_grid(source_image), global_step=self.global_step)
            self.logger.add_image("image/generate", make_grid(target_image), global_step=self.global_step)
        return [source_image, target_image], clip_loss

def make_grid(image):
    grid = torchvision.utils.make_grid(image, nrow=4, normalize=True, value_range=(-1, 1))
    return grid