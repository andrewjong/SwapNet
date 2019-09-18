from argparse import ArgumentParser

import torch
from torch import nn

import modules.loss
from datasets.data_utils import unnormalize, scale_tensor
from models.base_gan import BaseGAN
from modules.swapnet_modules import TextureModule
from util.decode_labels import decode_cloth_labels


class TextureModel(BaseGAN):
    """
    Implements training steps of the SwapNet Texture Module.
    """

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser = super(TextureModel, TextureModel).modify_commandline_options(
            parser, is_train
        )
        if is_train:
            parser.add_argument(
                "--lambda_l1",
                type=float,
                default=100,
                help="weight for L1 loss in final term",
            )
            parser.add_argument(
                "--lambda_feat",
                type=float,
                default=100,
                help="weight for feature loss in final term",
            )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # TODO: decode cloth visual
        self.visual_names = [
            # "textures_unnormalized",
            "cloths_decoded",
            # "DEBUG_random_input",
            "targets_unnormalized",
            "fakes",
            "fakes_scaled",
        ]

        if self.is_train:
            # Define additional loss for generator
            self.criterion_L1 = nn.L1Loss()
            self.criterion_features = modules.loss.get_vgg_feature_loss(opt, 1).to(
                self.device
            )

            self.loss_names = self.loss_names + ["G_l1", "G_feature"]

    def compute_visuals(self):
        self.textures_unnormalized = unnormalize(
            self.textures, *self.opt.texture_norm_stats
        )
        self.cloths_decoded = decode_cloth_labels(self.cloths)
        self.targets_unnormalized = unnormalize(
            self.targets, *self.opt.texture_norm_stats
        )

        self.fakes_scaled = scale_tensor(self.fakes, scale_each=True)
        # all batch, only first 3 channels
        # self.DEBUG_random_input = self.net_generator.DEBUG_random_input[:, :3] # take the top 3 layers, to 'sample' the RGB image

    def get_D_inchannels(self):
        return self.opt.texture_channels + self.opt.cloth_channels

    def define_G(self):
        return TextureModule(
            texture_channels=self.opt.texture_channels,
            cloth_channels=self.opt.cloth_channels,
            num_roi=self.opt.body_channels,
        )

    def set_input(self, input):
        textures, rois, cloths, targets = input
        self.textures = textures.to(self.device)
        self.rois = rois.to(self.device)
        self.cloths = cloths.to(self.device)
        self.targets = targets.to(self.device)

    def forward(self):
        self.fakes = self.net_generator(self.textures, self.rois, self.cloths)

    def backward_D(self):
        """
        Calculates loss and backpropagates for the discriminator
        """
        # https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L185
        if self.opt.gan_mode == "wgan":
            # clamp parameters to a cube
            for p in self.net_discriminator.parameters():
                p.data.clamp(-0.01, 0.01)

        # calculate fake
        fake_AB = torch.cat((self.cloths, self.fakes), 1)
        pred_fake = self.net_discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        real_AB = torch.cat((self.cloths, self.targets), 1)
        pred_real = self.net_discriminator(real_AB)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            # calculate gradient penalty
            self.loss_D_gp = modules.loss.gradient_penalty(
                self.net_discriminator, self.targets, self.fakes, self.opt.gan_mode
            )
            self.loss_D += self.opt.lambda_gp * self.loss_D_gp

        self.loss_D.backward()

    def backward_G(self):
        """
        Backward G for Texture stage.
        Loss composed of GAN loss, L1 loss, and feature loss.
        Returns:

        """
        fake_AB = torch.cat((self.cloths, self.fakes), 1)
        pred_fake = self.net_discriminator(fake_AB)
        self.loss_G_gan = self.criterion_GAN(pred_fake, True) * self.opt.lambda_gan

        self.loss_G_l1 = self.criterion_L1(self.fakes, self.targets) * self.opt.lambda_l1
        self.loss_G_feature = self.criterion_features(self.fakes, self.targets) * self.opt.lambda_feat

        # weighted sum
        self.loss_G = self.loss_G_gan + self.loss_G_l1 + self.loss_G_feature
        self.loss_G.backward()
