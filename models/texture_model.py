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
                default=1.0,
                help="weight for L1 loss in final term",
            )
            parser.add_argument(
                "--lambda_feat",
                type=float,
                default=1.0,
                help="weight for feature loss in final term",
            )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # TODO: decode cloth visual
        self.visual_names = [
            "textures_unnormalized",
            "cloths_decoded",
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

    def get_D_inchannels(self):
        return self.opt.texture_channels

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

    def backward_G(self):
        """
        Backward G for Texture stage.
        Loss composed of GAN loss, L1 loss, and feature loss.
        Returns:

        """
        pred_fake = self.net_discriminator(self.fakes)
        self.loss_G_gan = self.criterion_GAN(pred_fake, True)

        self.loss_G_l1 = self.criterion_L1(self.fakes, self.targets)
        self.loss_G_feature = self.criterion_features(self.fakes, self.targets)

        # weighted sum
        self.loss_G = (
            self.opt.lambda_gan * self.loss_G_gan
            + self.opt.lambda_l1 * self.loss_G_l1
            + self.opt.lambda_feat * self.loss_G_feature
        )
        self.loss_G.backward()
