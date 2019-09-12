import numpy as np
import torch
from PIL import Image

n_classes = 20
# colour map
label_colours = [(0,0,0)
                 # 0=Background
    ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                 # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
    ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                 # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
    ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                 # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
    ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
# 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe


# take out sunglasses
label_colours = label_colours[:4] + label_colours[5:]
n_classes = 19


def decode_cloth_labels(pt_tensor, num_images=-1, num_classes=n_classes):
    """Decode batch of segmentation masks.
    AJ comment: Converts the tensor into a RGB image.
    Args:
      as_tf_order: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    # change to H x W x C order
    tf_order = pt_tensor.permute(0, 2, 3, 1)
    argmax = tf_order.argmax(dim=-1, keepdim=True)
    mask = argmax.cpu().numpy()

    n, h, w, c = mask.shape
    if num_images < 0:
        num_images = n
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        # AJ: this enumerates the "rows" of the image (I think)
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < n_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(img)

    # convert back to tensor. effectively puts back into range [0,1]
    back_to_pt = torch.from_numpy(outputs).permute(0, 3, 1, 2)
    return back_to_pt
