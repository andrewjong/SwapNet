import copy
import json
import os
from argparse import Namespace

from tqdm import tqdm

from datasets import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images


def get_model_from_checkpoint(opt):
    model = create_model(opt)
    model.load_checkpoint(opt)


def run_warp(opt):
    warp_out = os.path.join(opt.results_dir, "warp")
    # create a copy of the config that loads in the config used for training. this is so
    # we load the model architecture
    warp_config = config.copy().load(
        os.path.join(config.opt.warp_checkpoint, "args.json")
    )
    warp_config.opt.from_epoch = opt.from_epoch
    warp_config.opt.is_train = False

    warp_model = create_model(warp_config.opt)
    warp_model.setup(warp_config.opt).eval()

    warp_dataset = create_dataset(
        warp_config.opt, body_dir=opt.body_dir, cloth_dir=opt.cloth_dir
    )

    # create a website
    web_dir = os.path.join(
        warp_out, f"inference_{opt.from_epoch}"
    )  # define the website directory
    webpage = html.HTML(
        web_dir,
        f"Experiment = {opt.name}, Phase = inference, Epoch = {opt.from_epoch}",
    )

    for i, data in tqdm(enumerate(warp_dataset), total=len(warp_dataset)):
        warp_model.set_input(data)
        warp_model.test()
        visuals = warp_model.get_current_visuals()
        image_paths = [f"_/{i}"]
        save_images(
            webpage,
            visuals,
            image_paths,
            width=opt.display_winsize,
        )
    webpage.save()  # save the HTML


if __name__ == "__main__":
    config = TestOptions(model="warp")
    config.parse()
    opt = config.opt
    if not opt.warp_checkpoint and not opt.texture_checkpoint:
        raise ValueError("Must set either warp_checkpoint or texture_checkpoint")
    opt.batch_size = 1

    if opt.warp_checkpoint:
        print("Runnng warp inference")
        run_warp(opt)

    if opt.texture_checkpoint:
        texture_model = get_model_from_checkpoint(opt.texture_checkpoint)
        texture_out = os.path.join(opt.results_dir, "texture")
