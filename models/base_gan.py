"""
A general framework for GAN training.
"""
from argparse import ArgumentParser
from abc import ABC, abstractmethod

import optimizers
from models import BaseModel
import modules.loss
from modules import discriminators
from modules.discriminators import Discriminator


class BaseGAN(BaseModel, ABC):
    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        """
        Adds several GAN-related training arguments.
        Child classes should call to extend this static method.
        >>> parser = super(ChildClass, ChildClass).modify_commandline_options(
        >>>    parser, is_train
        >>> )
        """
        if is_train:
            # gan mode choice
            parser.add_argument(
                "--gan_mode",
                help="gan regularization to use",
                default="vanilla",
                choices=(
                    "vanilla",
                    "wgan",
                    "wgan-gp",
                    "lsgan",
                    "dragan-gp",
                    "dragan-lp",
                    "mescheder-r1-gp",
                    "mescheder-r2-gp",
                ),
            )
            parser.add_argument(
                "--lambda_gan",
                type=float,
                default=1.0,
                help="weight for adversarial loss",
            )
            parser.add_argument(
                "--lambda_discriminator",
                type=float,
                default=1.0,
                help="weight for discriminator loss",
            )
            parser.add_argument(
                "--lambda_gp",
                help="weight parameter for gradient penalty",
                type=float,
                default=10,
            )
            # discriminator choice
            parser.add_argument(
                "--discriminator",
                default="basic",
                choices=("basic", "pixel", "n_layers"),
                help="what discriminator type to use",
            )
            parser.add_argument(
                "--n_layers_D",
                type=int,
                default=3,
                help="only used if discriminator==n_layers",
            )
            parser.add_argument(
                "--norm",
                type=str,
                default="instance",
                help="instance normalization or batch normalization [instance | batch | none]",
            )
            # optimizer choice
            parser.add_argument(
                "--optimizer_G",
                "--opt_G",
                "--optim_G",
                help="optimizer for generator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument(
                "--lr",
                "--g_lr",
                "--learning_rate",
                type=float,
                # as recommended by "10 Lessons I Learned Training GANs For a Year"
                default=0.0001,
                help="initial learning rate for generator",
            )
            parser.add_argument('--beta1', type=float, default=0.5,
                                help='momentum term of adam')
            parser.add_argument(
                "--optimizer_D",
                "--opt_D",
                "--optim_D",
                help="optimizer for discriminator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument(
                "--d_lr",
                type=float,
                # as recommended by "10 Lessons I Learned Training GANs For a Year"
                default=0.0004,
                help="initial learning rate for Discriminator",
            )
            parser.add_argument(
                "--d_wt_decay",
                "--d_weight_decay",
                dest="d_weight_decay",
                default=0.01,
                type=float,
                help="optimizer L2 weight decay",
            )
            parser.add_argument(
                "--gan_label_mode",
                default="smooth",
                choices=("hard", "smooth"),
                help="whether to use hard (real 1.0 and fake 0.0) or smooth "
                "(real [0.7, 1.1] and fake [0., 0.3]) values for labels",
            )
        return parser

    def __init__(self, opt):
        """
        Sets the generator, discriminator, and optimizers.

        Sets self.net_generator to the return value of self.define_G()

        Args:
            opt:
        """
        super().__init__(opt)
        self.net_generator = self.define_G().to(self.device)
        modules.init_weights(self.net_generator, opt.init_type, opt.init_gain)

        self.model_names = ["generator"]

        if self.is_train:
            # setup discriminator
            self.net_discriminator = discriminators.define_D(
                self.get_D_inchannels(), 64, opt.discriminator, opt.n_layers_D, opt.norm
            ).to(self.device)
            modules.init_weights(self.net_discriminator, opt.init_type, opt.init_gain)

            # load discriminator only at train time
            self.model_names.append("discriminator")

            # setup GAN loss
            use_smooth = True if opt.gan_label_mode == "smooth" else False
            self.criterion_GAN = modules.loss.GANLoss(
                opt.gan_mode, smooth_labels=use_smooth
            ).to(self.device)

            if opt.lambda_discriminator:
                self.loss_names = ["D", "D_real", "D_fake"]
                if any(gp_mode in opt.gan_mode for gp_mode in ["gp", "lp"]):
                    self.loss_names += ["D_gp"]
            self.loss_names += ["G"]
            if opt.lambda_gan:
                self.loss_names += ["G_gan"]

            # Define optimizers
            self.optimizer_G = optimizers.define_optimizer(
                self.net_generator.parameters(), opt, "G"
            )
            self.optimizer_D = optimizers.define_optimizer(
                self.net_discriminator.parameters(), opt, "D"
            )
            self.optimizer_names = ("G", "D")

    @abstractmethod
    def get_D_inchannels(self):
        """
        Return number of channels for discriminator input.
        Called when constructing the Discriminator network.
        """
        pass

    @abstractmethod
    def define_G(self):
        """
        Return the generator module. Called in init()
        The returned value is set to self.net_generator().
        """
        pass

    def optimize_parameters(self):
        self.forward()
        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

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
        pred_fake = self.net_discriminator(self.fakes.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # calculate real
        pred_real = self.net_discriminator(self.targets)
        self.loss_D_real = self.criterion_GAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_discriminator

        if any(gp_mode in self.opt.gan_mode for gp_mode in ["gp", "lp"]):
            # calculate gradient penalty
            self.loss_D_gp = modules.loss.gradient_penalty(
                self.net_discriminator, self.targets, self.fakes, self.opt.gan_mode
            )
            self.loss_D += self.opt.lambda_gp * self.loss_D_gp

        self.loss_D.backward()

    @abstractmethod
    def backward_G(self):
        """
        Calculate loss and backpropagates for the generator
        """
        pass
