"""
Custom loss modules
"""

import torch
from torch import nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    default_real = 1.0
    default_fake = 0
    default_smooth_real = (0.7, 1.1)
    default_smooth_fake = (0.0, 0.3)

    def __init__(
        self,
        gan_mode,
        smooth_labels=True,
        target_real_label=None,
        target_fake_label=None,
    ):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        if target_real_label is None:
            target_real_label = (
                self.default_smooth_real if smooth_labels else self.default_real
            )
        if target_fake_label is None:
            target_fake_label = (
                self.default_smooth_fake if smooth_labels else self.default_fake
            )

        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        # according to DRAGAN GitHub, dragan also uses BCE loss: https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        elif gan_mode in ["vanilla", "dragan", "dragan-gp", "dragan-lp"]:
            self.loss = nn.BCEWithLogitsLoss()
        elif "wgan" in gan_mode:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    @staticmethod
    def rand_between(low, high, normal=False):
        """
        Args:
            low: a torch.Tensor
            high: a torch.Tensor
            normal: whether to use normal distribution. if not, will use uniform

        Returns: random tensor between low and high
        """
        rand_func = torch.randn if normal else torch.rand
        return rand_func(1) * (high - low) + low

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            # smooth labels
            if len(self.real_label) == 2:
                low, high = self.real_label
                target_tensor = GANLoss.rand_between(low, high).to(
                    self.real_label.device
                )
            else:
                target_tensor = self.real_label
        else:
            # smooth labels
            if len(self.fake_label) == 2:
                low, high = self.real_label
                target_tensor = GANLoss.rand_between(low, high).to(
                    self.fake_label.device
                )
            else:
                target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla", "dragan-gp", "dragan-lp"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif "wgan" in self.gan_mode:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise ValueError(f"{self.gan_mode} not recognized")
        return loss


def gradient_penalty(f, real, fake, mode, p_norm=2):
    """
    From https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/torchprob/gan/loss.py
    Args:
        f: a discriminator
        real: target
        fake: generated
        mode: 
        p_norm: 

    Returns:

    """

    def _gradient_penalty(f, real, fake=None, penalty_type="gp", p_norm=2):
        def _interpolate(a, b=None):
            if b is None:  # interpolation in DRAGAN
                beta = torch.rand_like(a)
                b = a + 0.5 * a.std() * beta
            shape = [a.size(0)] + [1] * (a.dim() - 1)
            alpha = torch.rand(shape, device=a.device)
            inter = a + alpha * (b - a)
            return inter

        x = _interpolate(real, fake).detach()
        x.requires_grad = True
        pred = f(x)
        grad = torch.autograd.grad(
            pred, x, grad_outputs=torch.ones_like(pred), create_graph=True
        )[0]
        norm = grad.view(grad.size(0), -1).norm(p=p_norm, dim=1)

        if penalty_type == "gp":
            gp = ((norm - 1) ** 2).mean()
        elif penalty_type == "lp":
            gp = (torch.max(torch.zeros_like(norm), norm - 1) ** 2).mean()

        return gp

    if not mode or mode == "vanilla":
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    elif mode in ["dragan", "dragan-gp", "dragan-lp"]:
        penalty_type = "gp" if mode == "dragan" else mode[-2:]
        gp = _gradient_penalty(f, real, penalty_type=penalty_type, p_norm=p_norm)
    elif mode in ["wgan-gp", "wgan-lp"]:
        gp = _gradient_penalty(f, real, fake, penalty_type=mode[-2:], p_norm=p_norm)
    else:
        raise ValueError("Don't know how to handle gan mode", mode)

    # TODO: implement mescheder's simplified gradient penalties

    return gp


