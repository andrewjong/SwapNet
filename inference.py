import copy
import json
import os
from argparse import Namespace

from tqdm import tqdm

from datasets import create_dataset
from datasets.data_utils import compress_and_save_cloth, remove_extension
from models import create_model
from options.test_options import TestOptions
from util import html
from util.util import PromptOnce
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
    print(warp_dataset)

    # create a website
    web_dir = os.path.join(
        warp_out, f"inference_{opt.from_epoch}"
    )  # define the website directory
    if opt.compute_intermediates:
        webpage = html.HTML(
            web_dir,
            f"Experiment = {opt.name}, Phase = inference, Epoch = {opt.from_epoch}",
        )

    with tqdm(total=min(len(warp_dataset), opt.max_dataset_size), unit="img") as pbar:
        for i, data in enumerate(warp_dataset):
            if i > opt.max_dataset_size * opt.interval:
                break
            if i % opt.interval == 0:
                warp_model.set_input(data)
                warp_model.test()
                image_paths = warp_model.get_image_paths()
                if opt.compute_intermediates:
                    visuals = warp_model.get_current_visuals()
                    save_images(
                        webpage,
                        visuals,
                        image_paths,
                        width=opt.display_winsize,
                    )
                name = "_to_".join(
                    [remove_extension(os.path.basename(p)) for p in image_paths[0]])
                out_name = os.path.join(warp_out, name)
                compress_and_save_cloth(warp_model.fakes[0], out_name)

                pbar.update()

    if opt.compute_intermediates:
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
