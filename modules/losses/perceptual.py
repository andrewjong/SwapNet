import torch
import torch.nn as nn
from torchvision.models import vgg16


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


class PerceptualLoss(nn.Module):
    def __init__(self, normalize=True, use_style=False):
        """

        Args:
            normalize:
            use_style: whether to calculate style loss using gram matrix
        """
        super(PerceptualLoss, self).__init__()

        self.normalize = normalize
        self.use_style = use_style

        vgg = vgg16(pretrained=True).features

        slices_idx = [
            [0, 4],  # until 5th layer
            [4, 9],  # until 10th layer
            [9, 16],  # until 17th layer
            [16, 23],  # until 23rd layer
            [23, 30],  # until 31st layer
        ]

        self.net = torch.nn.Sequential()

        for i, idx in enumerate(slices_idx):
            seq = torch.nn.Sequential()
            for j in range(idx[0], idx[1]):
                seq.add_module(str(j), vgg[j])
            self.net.add_module(str(i), seq)

        for p in self.parameters():
            p.requires_grad = False

        self.mse = nn.MSELoss()

    def forward(self, output, target):

        output_f = self.get_features(output)
        with torch.no_grad():
            target_f = self.get_features(target)

        content_losses = []
        style_losses = []

        for o, t in zip(output_f, target_f):
            content_losses.append(self.mse(o, t))
            if self.use_style:
                gram_output = gram_matrix(output)
                gram_target = gram_matrix(target)
                style_losses.append(self.mse(gram_output, gram_target))
        content_loss = sum(content_losses)
        style_loss = sum(style_losses)
        return content_loss, style_loss

    def get_features(self, x):
        """Assumes x in [0, 1]: transform to [-1, 1]."""
        x = 2.0 * x - 1.0
        feats = []
        for i, s in enumerate(self.net):
            x = s(x)
            if self.normalize:  # unit L2 norm over features, this implies the loss is a cosine loss in feature space
                f = x / (torch.sqrt(torch.pow(x, 2).sum(1, keepdim=True)) + 1e-8)
            else:
                f = x
            feats.append(f)
        return feats
