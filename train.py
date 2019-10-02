"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from tqdm import tqdm
import time
from options.train_options import TrainOptions
from datasets import create_dataset
from models import create_model

from util.visualizer import Visualizer

print = tqdm.write

if __name__ == "__main__":
    config = TrainOptions()
    opt = config.parse()  # get training options
    if opt.config_file:
        config.load(opt.config_file)
    config.print(), config.save()
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    total_iters = 0  # the total number of training iterations

    # outer loop for different epochs;
    # we save the model by # <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in tqdm(range(opt.from_epoch + 1, opt.n_epochs + 1)):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0

        with tqdm(total=len(dataset), unit="image") as pbar:
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                # calculate loss functions, get gradients, update network weights
                model.optimize_parameters()

                if total_iters % opt.display_freq == 0:
                    # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(
                        model.get_current_visuals(), epoch, save_result
                    )

                if total_iters % opt.print_freq == 0:
                    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(
                        epoch, epoch_iter, losses, t_comp, t_data
                    )
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(
                            epoch, float(epoch_iter) / dataset_size, losses
                        )

                if opt.latest_checkpoint_freq and total_iters % opt.latest_checkpoint_freq == 0:
                    # cache our latest model every <save_latest_freq> iterations
                    print(
                        f"saving the latest model (epoch {epoch:d}, total_iters {total_iters:d}) "
                    )
                    save_suffix = "iter_%d" % total_iters if opt.save_by_iter else f"latest"
                    model.save_checkpoint(save_suffix)

                iter_data_time = time.time()
                pbar.update(len(data[0]))

        if opt.checkpoint_freq and epoch % opt.checkpoint_freq == 0:
            # cache our model every <save_epoch_freq> epochs
            print(
                f"saving the model at the end of epoch {epoch:d}, iters {total_iters:d}"
            )
            model.save_checkpoint("latest")
            model.save_checkpoint(epoch)


        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))
        # model.update_learning_rate()                     # update learning rates at the end of every epoch.
