from argparse import ArgumentParser

from torch import nn

import modules.loss
from models.base_gan import BaseGAN
from modules.swapnet_modules import TextureModule


class TextureModel(BaseGAN):
    """
    Implements training steps of the SwapNet Texture Module.
    """
    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        parser = super(TextureModel, TextureModel).modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument("--lambda_l1", help="weight for L1 loss in final term")
            parser.add_argument(
                "--lambda_feat", help="weight for feature loss in final term"
            )
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        if self.is_train:
            # Define additional loss for generator
            self.criterion_L1 = nn.L1Loss()
            self.criterion_features = modules.loss.get_vgg_feature_loss(opt, 1).to(
                self.device
            )

            self.loss_names = self.loss_names + ("G_l1", "G_feature")

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
        rois = TextureModule.reshape_rois(rois)
        self.textures = textures
        self.rois = rois
        self.cloths = cloths
        self.targets = targets

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
            self.lambda_gan * self.loss_G_gan
            + self.lambda_l1 * self.loss_G_l1
            + self.lambda_feat * self.loss_G_feature
        )
        self.loss_G.backward()
