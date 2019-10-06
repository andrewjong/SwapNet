import numpy as np
from PIL import Image, ImageDraw
import torch
import seaborn

from util.util import tensor2im

NUM_BODY_LABELS = 12

BODY_COLORS = (np.array(seaborn.color_palette("hls", NUM_BODY_LABELS)) * 255).astype(
    np.uint8
)


# TODO move to util file
def draw_rois_on_texture(
    rois: torch.Tensor, texture_tensors: torch.Tensor, width_factor=0.01
):
    """

    Args:
        rois: roi in
        texture_tensors:
        width:

    Returns:

    """

    samples = []

    # do for all in the batch
    for roi_batch, t in zip(rois, texture_tensors):
        # unsqueeze because of batch annoyances
        im = Image.fromarray(tensor2im(t.unsqueeze(0)))
        draw = ImageDraw.Draw(im)
        for i, roi_row in enumerate(roi_batch.cpu()):
            draw.rectangle(
                roi_row.numpy(),
                outline=tuple(BODY_COLORS[i]),
                width=int(round(width_factor * im.size[0])),
            )

        samples.append(np.array(im))

    # return to batch size
    return np.stack(samples)
