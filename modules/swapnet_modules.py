"""
Networks based on SwapNet (Raj et al. 2018).
"""
import code
import logging
import sys
import math

import torch

from modules import pix2pix_modules, get_norm_layer

sys.path.append("../lib")
from torchvision.ops import RoIAlign as ROIAlign
from torch import nn

from modules.layers import UNetDown, UNetUp, DualUNetUp, ResidualBlock

wm_log = logging.getLogger("warp_module_shape")


class WarpModule(nn.Module):
    """
    The warping module takes a body segmentation to represent the "pose",
    and an input clothing segmentation to transform to match the pose.
    """

    def __init__(self, body_channels=3, cloth_channels=19, dropout=0.5):
        super(WarpModule, self).__init__()

        ######################
        # Body pre-encoding  #  (top left of SwapNet diagram)
        ######################
        self.body_down1 = UNetDown(body_channels, 64, normalize=False)
        self.body_down2 = UNetDown(64, 128)
        self.body_down3 = UNetDown(128, 256)
        self.body_down4 = UNetDown(256, 512, dropout=dropout)

        ######################
        # Cloth pre-encoding #  (bottom left of SwapNet diagram)
        ######################
        self.cloth_down1 = UNetDown(cloth_channels, 64, normalize=False)
        self.cloth_down2 = UNetDown(64, 128)
        self.cloth_down3 = UNetDown(128, 256)
        self.cloth_down4 = UNetDown(256, 512)
        self.cloth_down5 = UNetDown(512, 1024, dropout=dropout)
        self.cloth_down6 = UNetDown(1024, 1024, normalize=False, dropout=dropout)
        # the two UNetUp's below will be used WITHOUT concatenation.
        # hence the input size will not double
        self.cloth_up1 = UNetUp(1024, 1024)
        self.cloth_up2 = UNetUp(1024, 512)

        ######################
        #      Resblocks     #  (middle of SwapNet diagram)
        ######################
        self.resblocks = nn.Sequential(
            # I don't really know if dropout should go here. I'm just guessing
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
            ResidualBlock(1024, dropout=dropout),
        )

        ######################
        #    Dual Decoding   #  (right of SwapNet diagram, maybe)
        ######################
        # The SwapNet diagram just says "cloth" decoder, so I don't know if they're
        # actually doing dual decoding like I've done here.
        # Still, I think it's cool and it makes more sense to me.
        # Found from "Multi-view Image Generation from a Single-View".
        # ---------------------
        # input encoded (512) & cat body_d4 (512) cloth_d4 (512)
        self.dual_up1 = DualUNetUp(1024, 256)
        # input dual_up1 (256) & cat body_d3 (256) cloth_d3 (256)
        self.dual_up2 = DualUNetUp(3 * 256, 128)
        # input dual_up2 (128) & cat body_d2 (128) cloth_d2 (128)
        self.dual_up3 = DualUNetUp(3 * 128, 64)

        # TBH I don't really know what the below code does.
        # like why don't we dualnetup with down1?
        # maybe specific to pix2pix? hm, if so maybe we should replicate.
        # ------
        # update: OHHH I get it now. it's because U-Net only outputs half the size as
        #  the original image, hence we need to upsample.
        self.upsample_and_pad = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(3 * 64, cloth_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, body, cloth):
        wm_log.debug("body shape:", body.shape)
        wm_log.debug("cloth shape:", cloth.shape)
        wm_log.debug("shapes should match except in the channel dim")
        ######################
        # Body pre-encoding  #
        ######################
        body_d1 = self.body_down1(body)
        body_d2 = self.body_down2(body_d1)
        wm_log.debug("body_d2 shape, should be 128 channel:", body_d2.shape)
        body_d3 = self.body_down3(body_d2)
        wm_log.debug("body_d3 shape, should be 256 channel:", body_d3.shape)
        body_d4 = self.body_down4(body_d3)
        wm_log.debug("body_d4 shape, should be 512 channel:", body_d4.shape)

        wm_log.debug("==============")
        ######################
        # Cloth pre-encoding #
        ######################
        cloth_d1 = self.cloth_down1(cloth)
        cloth_d2 = self.cloth_down2(cloth_d1)
        wm_log.debug("cloth_d2 shape, should be 128 channel:", cloth_d2.shape)
        cloth_d3 = self.cloth_down3(cloth_d2)
        wm_log.debug("cloth_d3 shape, should be 256 channel:", cloth_d3.shape)
        cloth_d4 = self.cloth_down4(cloth_d3)
        wm_log.debug("cloth_d4 shape, should be 512 channel:", cloth_d4.shape)
        cloth_d5 = self.cloth_down5(cloth_d4)
        wm_log.debug("cloth_d5 shape, should be 1024 channel:", cloth_d5.shape)
        cloth_d6 = self.cloth_down6(cloth_d5)
        wm_log.debug("cloth_d6 shape, should be 1024 channel:", cloth_d6.shape)
        cloth_u1 = self.cloth_up1(cloth_d6, None)
        wm_log.debug("cloth_u1 shape, should be 1024 channel:", cloth_u1.shape)
        cloth_u2 = self.cloth_up2(cloth_u1, None)
        wm_log.debug("cloth_u2 shape, should be 512 channel:", cloth_u2.shape)

        #######################
        # Combine & Resblocks #
        #######################
        # cat on the channel dimension? should be same HxW
        body_and_cloth = torch.cat((body_d4, cloth_u2), dim=1)
        wm_log.debug(
            "body_and_cloth shape, should be 1024 channel:", body_and_cloth.shape
        )
        encoded = self.resblocks(body_and_cloth)
        wm_log.debug("encoded shape, should be 1024 channel:", encoded.shape)

        # ######################
        # #    Dual Decoding   #
        # ######################
        dual_u1 = self.dual_up1(encoded, body_d3, cloth_d3)
        wm_log.debug("dual_u1 shape, should be 3*256 channel:", dual_u1.shape)
        dual_u2 = self.dual_up2(dual_u1, body_d2, cloth_d2)
        wm_log.debug("dual_u2 shape, should be 3*128 channel:", dual_u2.shape)
        dual_u3 = self.dual_up3(dual_u2, body_d1, cloth_d1)
        wm_log.debug("dual_u3 shape, should be 3*64 channel:", dual_u3.shape)

        # this is from that commented out code in the __init__()
        upsampled = self.upsample_and_pad(dual_u3)
        wm_log.debug("upsampled shape, should be original channel:", upsampled.shape)
        return upsampled


class TextureModule(nn.Module):
    def __init__(
        self,
        texture_channels=3,
        cloth_channels=19,
        num_roi=12,
        norm_type="batch",
        dropout=0.5,
        unet_type="pix2pix",
        img_size=128,
    ):
        super(TextureModule, self).__init__()
        self.roi_align = ROIAlign(
            output_size=(128, 128), spatial_scale=1, sampling_ratio=1
        )

        self.num_roi = num_roi
        channels = texture_channels * num_roi
        self.encode = UNetDown(channels, channels)

        # UNET

        if unet_type == "pix2pix":
            # fast log2 of img_size, int only. E.g. if size=128 => num_downs=7
            num_downs = math.frexp(img_size)[1] - 1
            use_dropout = True if dropout is not None else False
            norm_layer = get_norm_layer(norm_type=norm_type)
            self.unet = pix2pix_modules.UnetGenerator(
                channels + cloth_channels,
                texture_channels,
                num_downs,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        else:
            self.unet = nn.Sequential(
                UNetDown(channels + cloth_channels, 64, normalize=False),
                UNetDown(64, 128),
                UNetDown(128, 256),
                UNetDown(256, 512, dropout=dropout),
                UNetDown(512, 1024, dropout=dropout),
                UNetDown(1024, 1024, normalize=False, dropout=dropout),
                UNetUp(1024, 1024, dropout=dropout),
                UNetUp(2 * 1024, 512, dropout=dropout),
                UNetUp(2 * 512, 256),
                UNetUp(2 * 256, 128),
                UNetUp(2 * 128, 64),
                # upsample and pad
                nn.Upsample(scale_factor=2),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(128, texture_channels, 4, padding=1),
                nn.Tanh(),
            )
        # print("u-net:", self.unet)

    @staticmethod
    def reshape_rois(rois):
        """
        Takes a (batch x num_rois x num_coordinates) and reshapes it into a 2D tensor.
        The 2D tensor has the first column as the batch index and the remaining columns
        as the coordinates.

        num_coordinates should be 4.

        :param rois: a (batch x num_rois x num_coordinates) tensor. coordinates is 4
        :return: a 2D tensor formatted for roi layers
        """
        # Append batch to rois
        # get the batch indices
        b_idx = torch.arange(rois.shape[0]).unsqueeze_(-1)
        # expand out and reshape to to batchx1 dimension
        b_idx = b_idx.expand(rois.shape[0], rois.shape[1]).reshape(-1).unsqueeze_(-1)
        b_idx = b_idx.to(rois.device).type(rois.dtype)
        reshaped = rois.view(-1, rois.shape[-1])
        reshaped = torch.cat((b_idx, reshaped), dim=1)
        return reshaped

    def forward(self, input_tex, rois, cloth):
        rois = TextureModule.reshape_rois(rois)
        # do roi alignment
        pooled_rois = self.roi_align(input_tex, rois)
        # reshape the pooled rois such that pool output goes in the channels instead of
        # batch size
        batch_size = int(pooled_rois.shape[0] / self.num_roi)
        pooled_rois = pooled_rois.view(
            batch_size, -1, pooled_rois.shape[2], pooled_rois.shape[3]
        )

        encoded_tex = self.encode(pooled_rois)

        scale_factor = input_tex.shape[2] / encoded_tex.shape[2]
        upsampled_tex = nn.functional.interpolate(
            encoded_tex, scale_factor=scale_factor
        )

        # # DEBUG DEBUG: see if GAN works without textured input
        # s = torch.sum(upsampled_tex).item()
        # if s != 0:
        #     print(
        #         f"Warning!!! Input of 0 did not result in upsampled_tex of 0. Sum is {s}. Setting to 0s anyway"
        #     )
        # upsampled_tex = torch.zeros_like(upsampled_tex)

        # concat on the channel dimension
        tex_with_cloth = torch.cat((upsampled_tex, cloth), 1)

        return self.unet(tex_with_cloth)
