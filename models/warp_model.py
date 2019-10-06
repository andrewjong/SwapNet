from argparse import ArgumentParser

import torch
from torch import nn

import modules.loss
from datasets.data_utils import unnormalize, remove_top_dir
from models import BaseModel
from models.base_gan import BaseGAN
from modules.swapnet_modules import WarpModule
from util.decode_labels import decode_cloth_labels


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
            parser.add_argument("--warp_mode", default="gan", choices=("gan", "ce"))
            parser.add_argument(
                "--lambda_ce",
                type=float,
                default=100,
                help="weight for cross entropy loss in final term",
            )
            # based on the num entries in self.visual_names during training
            parser.set_defaults(display_ncols=4)
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(WarpModel, WarpModel).modify_commandline_options(
            parser, is_train
        )
        return parser

    def __init__(self, opt):
        """
        Initialize the WarpModel. Either in GAN mode or plain Cross Entropy mode.
        Args:
            opt:
        """
        # 3 for RGB
        self.body_channels = (
            opt.body_channels if opt.body_representation == "labels" else 3
        )
        # 3 for RGB
        self.cloth_channels = (
            opt.cloth_channels if opt.cloth_representation == "labels" else 3
        )

        BaseGAN.__init__(self, opt)

        # TODO: decode visuals for cloth
        self.visual_names = ["inputs_decoded", "bodys_unnormalized", "fakes_decoded"]

        if self.is_train:
            self.visual_names.append(
                "targets_decoded"
            )  # only show targets during training
            # we use cross entropy loss in both
            self.criterion_CE = nn.CrossEntropyLoss()
            if opt.warp_mode != "gan":
                # remove discriminator related things if no GAN
                self.model_names = ["generator"]
                self.loss_names = "G"
                del self.net_discriminator
                del self.optimizer_D
                self.optimizer_names = ["G"]
            else:
                self.loss_names += ["G_ce"]

    def compute_visuals(self):
        self.inputs_decoded = decode_cloth_labels(self.inputs)
        self.bodys_unnormalized = unnormalize(self.bodys, *self.opt.body_norm_stats)
        self.targets_decoded = decode_cloth_labels(self.targets)
        self.fakes_decoded = decode_cloth_labels(self.fakes)

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
        self.bodys = input["bodys"].to(self.device)
        self.inputs = input["input_cloths"].to(self.device)
        self.targets = input["target_cloths"].to(self.device)

        self.image_paths = tuple(zip(input["cloth_paths"], input["body_paths"]))

    def forward(self):
        self.fakes = self.net_generator(self.bodys, self.inputs)

    def backward_D(self):
        """
        Warp stage's custom backward_D implementation passes CONDITIONED input to 
        the discriminator. Concats the bodys with the cloth
        """
        # calculate fake
        conditioned_fake = torch.cat((self.bodys, self.fakes), 1)
        pred_fake = self.net_discriminator(conditioned_fake.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        conditioned_real = torch.cat((self.bodys, self.targets), 1)
        pred_real = self.net_discriminator(conditioned_real)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)

        # calculate gradient penalty
        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            self.loss_D_gp = (
                modules.loss.gradient_penalty(
                    self.net_discriminator,
                    conditioned_real,
                    conditioned_fake,
                    self.opt.gan_mode,
                )
                * self.opt.lambda_gp
            )
            self.loss_D += self.loss_D_gp

        # final loss
        self.loss_D.backward()

    def backward_G(self):
        """
        If GAN mode, loss is weighted sum of cross entropy loss and adversarial GAN
        loss. Else, loss is just cross entropy loss.
        """
        # cross entropy loss needed for both gan mode and ce mode
        loss_ce = (
            self.criterion_CE(self.fakes, torch.argmax(self.targets, dim=1))
            * self.opt.lambda_ce
        )

        # if we're in GAN mode, calculate adversarial loss too
        if self.opt.warp_mode == "gan":
            self.loss_G_ce = loss_ce  # store loss_ce

            # calculate adversarial loss
            conditioned_fake = torch.cat((self.bodys, self.fakes), 1)
            pred_fake = self.net_discriminator(conditioned_fake)
            self.loss_G_gan = self.criterion_GAN(pred_fake, True) * self.opt.lambda_gan

            # total loss is weighted sum
            self.loss_G = self.loss_G_gan + self.loss_G_ce
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
