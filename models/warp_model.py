from argparse import ArgumentParser

import torch
from torch import nn

import modules.loss
from models import BaseModel
from models.base_gan import BaseGAN
from modules.swapnet_modules import WarpModule


class WarpModel(BaseGAN):
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
            parser.set_defaults(lambda_gan=0.1) # swapnet says "*small* adversarial loss"
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(WarpModel, WarpModel).modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """
        Initialize the WarpModel. Either in GAN mode or plain Cross Entropy mode.
        Args:
            opt:
        """
        self.body_channels = opt.body_channels if opt.body_representation == "labels" else 3 # 3 for RGB
        self.cloth_channels = opt.cloth_channels if opt.cloth_representation == "labels" else 3 # 3 for RGB

        BaseGAN.__init__(self, opt)


        if self.is_train:
            # we use cross entropy loss in both
            self.criterion_CE = nn.CrossEntropyLoss()
            if opt.warp_mode != "gan":
                # remove discriminator related things
                self.model_names = ["generator"]
                self.loss_names = ("G")
                del self.net_discriminator
                del self.optimizer_D
                self.optimizer_names = ["G"]

    def define_G(self):
        """
        The generator is the Warp Module.
        """
        return WarpModule(
            body_channels=self.body_channels, cloth_channels=self.cloth_channels
        )

    def get_D_inchannels(self):
        """
        The Warp stage discriminator is a conditional discriminator. 
        This means we concatenate the generated warped cloth with the body segmentation.
        """
        return self.cloth_channels + self.body_channels

    def set_input(self, input):
        bodys, inputs, targets = input
        self.bodys = bodys
        self.inputs = inputs
        self.targets = targets

    def forward(self):
        self.fakes = self.net_generator(self.bodys, self.inputs)

    def backward_D(self):
        """
        Warp stage's custom backward_D implementation passes CONDITIONED input to 
        the discriminator. Concats the bodys with the cloth
        """
        # calculate real
        # THIS LINE:
        conditioned_fake_input = torch.cat((self.fakes.detach(), self.bodys), 1)
        pred_fake = self.net_discriminator(conditioned_fake_input)
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate fake
        # AND THIS LINE ARE THE ONLY TWO CHANGED:
        conditioned_real_input = torch.cat((self.targets, self.bodys), 1)
        pred_real = self.net_discriminator(conditioned_real_input)
        self.loss_D_real = self.criterion_GAN(pred_real, True)
        # calculate gradient penalty
        self.loss_D_gp = modules.loss.gradient_penalty(
            self.net_discriminator, self.targets, self.fakes, self.opt.gan_mode
        )
        # final loss
        self.loss_D = (
            0.5 * (self.loss_D_fake + self.loss_D_real)
            + self.opt.lambda_gp * self.loss_D_gp
        )
        self.loss_D.backward()

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
            conditioned_fake = torch.cat((self.fakes, self.bodys), 1)
            pred_fake = self.net_discriminator(conditioned_fake)
            self.loss_G_gan = self.criterion_GAN(pred_fake, True)

            # total loss is weighted sum
            self.loss_G = self.opt.lambda_gan * self.loss_G_gan + self.loss_G_ce
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
            self.forward()
            # optimize G only
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
