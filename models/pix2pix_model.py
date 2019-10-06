from datasets.data_utils import unnormalize, scale_tensor
from models import BaseModel
import torch

from modules.discriminators import define_D
from modules.loss import GANLoss
from modules.pix2pix_modules import define_G
from util.decode_labels import decode_cloth_labels


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128')
        if is_train:
            parser.add_argument('--lambda_l1', type=float, default=10, help='weight for L1 loss')
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
                help="optimizer for generator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument(
                "--optimizer_D",
                help="optimizer for discriminator",
                default="AdamW",
                choices=("AdamW", "AdaBound"),
            )
            parser.add_argument('--beta1', type=float, default=0.5,
                                help='momentum term of adam')
            parser.add_argument(
                "--gan_label_mode",
                default="smooth",
                choices=("hard", "smooth"),
                help="whether to use hard (real 1.0 and fake 0.0) or smooth "
                     "(real [0.7, 1.1] and fake [0., 0.3]) values for labels",
            )
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser


    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['cloth_decoded', 'fakes_scaled', 'textures_unnormalized']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.net_G = define_G(opt.cloth_channels + 36, opt.texture_channels, 64, "unet_128", opt.norm, True, opt.init_type, opt.init_gain).to(self.device)

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.net_D = define_D(opt.cloth_channels + 36 + opt.texture_channels, 64, opt.discriminator, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain).to(self.device)

        if self.is_train:
            # define loss functions
            use_smooth = True if opt.gan_label_mode == "smooth" else False
            self.criterionGAN = GANLoss(opt.gan_mode, smooth_labels=use_smooth).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        to_concat = torch.zeros((self.opt.batch_size, 36, self.opt.crop_size, self.opt.crop_size), device=self.device)
        self.real_A = torch.cat((to_concat, input["cloths"].to(self.device)), 1)

        # self.real_A = torch.randn_like(cloth_tensor).to(self.device)
        self.real_B = input["target_textures"].to(self.device)

    def compute_visuals(self):
        self.cloth_decoded = decode_cloth_labels(self.real_A)
        self.fakes_scaled = scale_tensor(self.fake_B)
        self.textures_unnormalized = unnormalize(
            self.real_B, *self.opt.texture_norm_stats
        )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.net_G(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.net_D(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.net_D(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.net_D(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_l1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.net_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.net_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
