from argparse import ArgumentParser

import torch
from torch import nn

from models import BaseModel
from models.base_gan import BaseGAN
from modules.swapnet_modules import WarpModule


class WarpModel(BaseModel, BaseGAN):
    """
    Implements training steps of the SwapNet Texture Module.
    """
    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        """
        Adds warp_mode option for generator loss. This is because Khiem found out using
        plain Cross Entropy works just fine. CE mode saves time and space by not having
        to train an additional discriminator network.
        """
        if is_train:
            parser.add_argument("--warp_mode", choices=("gan", "ce"))
        # TODO. this line is going to add GAN options regardless right? how do we prevent that
        parser = super().modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """
        Initialize the WarpModel. Either in GAN mode or plain Cross Entropy mode.
        Args:
            opt:
        """
        # whether we're going to use GAN for training warp or not
        if opt.warp_mode == "gan":
            BaseGAN.__init__(self, opt)
        else:
            BaseModel.__init__(self, opt)
            self.generator = self.define_G()

        if self.is_train:
            # we use cross entropy loss in both
            self.criterion_CE = nn.CrossEntropyLoss()
            if opt.warp_mode == "gan":
                self.loss_names = super().loss_names + ("G_gan", "G_ce")

    def define_G(self):
        """
        The generator is the Warp Module.
        """
        return WarpModule(
            body_channels=self.opt.body_channels, cloth_channels=self.opt.cloth_channels
        )

    def get_D_inchannels(self):
        """
        The discriminator is deciding between cloth segmentations, so we return number
        of cloth channels.
        """
        return self.opt.cloth_channels

    def set_input(self, input):
        bodys, inputs, targets = input
        self.bodys = bodys
        self.inputs = inputs
        self.targets = targets

    def forward(self):
        self.fakes = self.generator(self.bodys, self.inputs)

    def backward_G(self):
        """
        If GAN mode, loss is weighted sum of cross entropy loss and adversarial GAN
        loss. Else, loss is just cross entropy loss.
        """
        # cross entropy loss needed for both gan mode and ce mode
        loss_ce = self.criterion_CE(self.fakes, torch.argmax(self.targets, dim=1))

        # if we're in GAN mode, calculate adversarial loss too
        if self.opt.warp_mode == "gan":
            self.loss_G_ce = loss_ce  # store loss_ce

            # calculate adversarial loss
            pred_fake = self.discriminator(self.fake_tex)
            self.loss_G_gan = self.criterion_GAN(pred_fake, True)

            # total loss is weighted sum
            self.loss_G = self.lambda_gan * self.loss_G_gan + self.loss_G_ce
        else:
            # otherwise our only loss is cross entropy
            self.loss_G = loss_ce

        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Optimize both G and D if in GAN mode, else just G.
        Returns:

        """
        if self.opt.warp_mode == "gan":
            # will optimize both D and G
            super().optimize_parameters()
        else:
            # optimize G only
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
