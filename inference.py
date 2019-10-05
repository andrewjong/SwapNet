import argparse
import os
from typing import Callable

from tqdm import tqdm

from datasets import create_dataset
from datasets.data_utils import compress_and_save_cloth, remove_extension
from models import create_model
from options.test_options import TestOptions
from util import html
from util.util import PromptOnce
from util.visualizer import save_images

WARP_SUBDIR = "warp"
TEXTURE_SUBDIR = "texture"

# FUNCTIONS SHOULD NOT BE IMPORTED BY OTHER MODULES. THEY ARE ONLY HELPER METHODS,
# AND DEPEND ON GLOBAL VARIABLES UNDER MAIN


def _setup(subfolder_name):
    """
    Setup outdir, create a webpage
    Args:
        subfolder_name: name of the outdir and where the webpage files should go

    Returns:

    """
    out_dir = get_out_dir(subfolder_name)
    PromptOnce.makedirs(out_dir, not opt.no_confirm)
    webpage = None
    if not opt.warp_checkpoint or opt.visualize_intermediates:
        webpage = html.HTML(
            out_dir,
            f"Experiment = {opt.name}, Phase = {subfolder_name} inference, "
            f"Loaded Epoch = {opt.from_epoch}",
        )
    return out_dir, webpage


def get_out_dir(subfolder_name):
    return os.path.join(opt.results_dir, subfolder_name)


def _rebuild_from_checkpoint(checkpoint_file, same_crop_load_size=False, **ds_kwargs):
    """
    Loads a model and dataset based on the config in a particular dir.
    Args:
        checkpoint_file: dir containing args.json and model checkpoints
        **ds_kwargs: override kwargs for dataset

    Returns: loaded model, initialized dataset

    """
    checkpoint_dir = os.path.dirname(checkpoint_file)
    loaded_opt = _copy_and_load_config(checkpoint_dir).opt
    # force certain attributes in the loaded cfg
    override_namespace(
        loaded_opt,
        is_train=False,
        batch_size=1,
        shuffle_data=opt.shuffle_data,  # let inference opt take precedence
    )
    if same_crop_load_size:  # need to override this if we're using intermediates
        loaded_opt.load_size = loaded_opt.crop_size
    model = create_model(loaded_opt)
    model.load_model_weights(
        "generator", checkpoint_file
    ).eval()  # loads the checkpoint
    model.print_networks(opt.verbose)

    dataset = create_dataset(loaded_opt, **ds_kwargs)

    return model, dataset


def _copy_and_load_config(directory):
    """
    Copies the global config and loads in train arguments from "args.json".

    This is so we can reconstruct the model/dataset.
    Args:
        directory: directory to load "args.json" from

    Returns:

    """
    return config.copy().load(os.path.join(directory, "args.json"))


def override_namespace(namespace, **kwargs):
    """
    Simply overrides the attributes in the object with the specified keyword arguments
    Args:
        namespace: argparse.Namespace object
        **kwargs: keyword/value pairs to use as override
    """
    assert isinstance(namespace, argparse.Namespace)
    for k, v in kwargs.items():
        setattr(namespace, k, v)


def _run_test_loop(model, dataset, webpage=None, iteration_post_hook: Callable = None):
    """

    Args:
        model: object that extends BaseModel
        dataset: object that extends BaseDataset
        webpage: webpage object for saving
        iteration_post_hook: a function to call at the end of every iteration

    Returns:

    """

    total = min(len(dataset), opt.max_dataset_size)
    with tqdm(total=total, unit="img") as pbar:
        for i, data in enumerate(dataset):
            if i >= total:
                break
            model.set_input(data)  # set input
            model.test()  # forward pass
            image_paths = model.get_image_paths()  # ids of the loaded images

            if webpage:
                visuals = model.get_current_visuals()
                save_images(webpage, visuals, image_paths, width=opt.display_winsize)

            if iteration_post_hook:
                iteration_post_hook(local=locals())

            pbar.update()

    if webpage:
        webpage.save()


def _run_warp():
    """
    Runs the warp stage
    """

    warp_out, webpage = _setup(WARP_SUBDIR)

    warp_model, warp_dataset = _rebuild_from_checkpoint(
        opt.warp_checkpoint, body_dir=opt.body_dir, cloth_dir=opt.cloth_dir
    )

    def save_cloths_npz(local):
        """
        We must store the intermediate cloths as .npz files
        """
        name = "_to_".join(
            [remove_extension(os.path.basename(p)) for p in local["image_paths"][0]]
        )
        out_name = os.path.join(warp_out, name)
        # save the warped cloths
        compress_and_save_cloth(local["model"].fakes[0], out_name)

    _run_test_loop(
        warp_model, warp_dataset, webpage, iteration_post_hook=save_cloths_npz
    )


def _run_texture():
    """
    Runs the texture stage. If opt.warp_checkpoint is also True, then it will use those
    intermediate cloth outputs as the texture stage's input.
    """
    _, webpage = _setup(TEXTURE_SUBDIR)

    if opt.warp_checkpoint:  # if intermediate, cloth dir is the warped cloths
        cloth_dir = get_out_dir(WARP_SUBDIR)
    else:  # otherwise if texture checkpoint alone, use what the user specified
        cloth_dir = opt.cloth_dir

    texture_model, texture_dataset = _rebuild_from_checkpoint(
        opt.texture_checkpoint,
        same_crop_load_size=True if opt.warp_checkpoint else False,
        texture_dir=opt.texture_dir,
        cloth_dir=cloth_dir,
    )

    _run_test_loop(texture_model, texture_dataset, webpage)


if __name__ == "__main__":
    config = TestOptions()
    config.parse()
    opt = config.opt

    if opt.warp_checkpoint:
        print("Running warp inference...")
        _run_warp()

    if opt.texture_checkpoint:
        print("Running texture inference...")
        _run_texture()

    print("\nDone!")
