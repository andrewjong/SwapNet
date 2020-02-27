import argparse
import copy
import os
from typing import Callable

from tqdm import tqdm

from datasets import create_dataset
from datasets.data_utils import compress_and_save_cloth, remove_extension
from models import create_model
from options.base_options import load
from options.test_options import TestOptions
from util import html
from util.util import PromptOnce
from util.visualizer import save_images

WARP_SUBDIR = "warp"
TEXTURE_SUBDIR = "texture"


# FUNCTIONS SHOULD NOT BE IMPORTED BY OTHER MODULES. THEY ARE ONLY HELPER METHODS,
# AND DEPEND ON GLOBAL VARIABLES UNDER MAIN


def _setup(subfolder_name, create_webpage=True):
    """
    Setup outdir, create a webpage
    Args:
        subfolder_name: name of the outdir and where the webpage files should go

    Returns:

    """
    out_dir = get_out_dir(subfolder_name)
    PromptOnce.makedirs(out_dir, not opt.no_confirm)
    webpage = None
    if create_webpage:
        webpage = html.HTML(
            out_dir,
            f"Experiment = {opt.name}, Phase = {subfolder_name} inference, "
            f"Loaded Epoch = {opt.load_epoch}",
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
    # read the config file  so we can load in the model
    loaded_opt = load(copy.deepcopy(opt), os.path.join(checkpoint_dir, "args.json"))
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
    # loads the checkpoint
    model.load_model_weights("generator", checkpoint_file).eval()
    model.print_networks(opt.verbose)

    dataset = create_dataset(loaded_opt, **ds_kwargs)

    return model, dataset


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
    warp_out, webpage = _setup(WARP_SUBDIR, create_webpage=not opt.skip_intermediates)

    print(f"Rebuilding warp from {opt.warp_checkpoint}")
    warp_model, warp_dataset = _rebuild_from_checkpoint(
        opt.warp_checkpoint, cloth_dir=opt.cloth_dir, body_dir=opt.body_dir
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

    print(f"Warping cloth to match body segmentations in {opt.body_dir}...")

    try:
        _run_test_loop(
            warp_model, warp_dataset, webpage, iteration_post_hook=save_cloths_npz
        )
    except KeyboardInterrupt:
        print("Ending warp early.")
    print(f"Warp results stored in {warp_out}")


def _run_texture():
    """
    Runs the texture stage. If opt.warp_checkpoint is also True, then it will use those
    intermediate cloth outputs as the texture stage's input.
    """
    texture_out, webpage = _setup(TEXTURE_SUBDIR, create_webpage=True)

    if opt.warp_checkpoint:  # if intermediate, cloth dir is the warped cloths
        cloth_dir = get_out_dir(WARP_SUBDIR)
    else:  # otherwise if texture checkpoint alone, use what the user specified
        cloth_dir = opt.cloth_dir

    print(f"Rebuilding texture from {opt.texture_checkpoint}")
    texture_model, texture_dataset = _rebuild_from_checkpoint(
        opt.texture_checkpoint,
        same_crop_load_size=True if opt.warp_checkpoint else False,
        texture_dir=opt.texture_dir,
        cloth_dir=cloth_dir,
    )

    print(f"Texturing cloth segmentations in {cloth_dir}...")
    try:
        _run_test_loop(texture_model, texture_dataset, webpage)
    except KeyboardInterrupt:
        print("Ending texture early.")
    print(f"Textured results stored in {texture_out}")


if __name__ == "__main__":
    config = TestOptions()
    config.parse()
    opt = config.opt

    # override checkpoint options
    if opt.checkpoint:
        if not opt.warp_checkpoint:
            opt.warp_checkpoint = os.path.join(
                opt.checkpoint, "warp", f"{opt.load_epoch}_net_generator.pth"
            )
            print("Set warp_checkpoint to", opt.warp_checkpoint)
        if not opt.texture_checkpoint:
            opt.texture_checkpoint = os.path.join(
                opt.checkpoint, "texture", f"{opt.load_epoch}_net_generator.pth"
            )
            print("Set texture_checkpoint to", opt.texture_checkpoint)

    # use dataroot if not individually provided
    for subdir in ("body", "cloth", "texture"):
        attribute = f"{subdir}_dir"
        if not getattr(opt, attribute):
            setattr(opt, attribute, os.path.join(opt.dataroot, subdir))

    # Run warp stage
    if opt.warp_checkpoint:
        print("Running warp inference...")
        _run_warp()

    # Run texture stage
    if opt.texture_checkpoint:
        print("Running texture inference...")
        _run_texture()

    print("\nDone!")
