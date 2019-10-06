from argparse import ArgumentParser

import torch
from adabound import AdaBound
from torch.optim import AdamW

AdamW, AdaBound


def get_options_modifier(optimizer_name):
    optimizer_name = optimizer_name.lower()

    # Adam or AdamW
    if "adam" in optimizer_name.lower():
        return adam_modifier
    # AdaBound or AdaBoundW
    elif "adabound" in optimizer_name.lower():
        return adabound_modifier
    elif "sgd" in optimizer_name.lower():
        raise NotImplementedError
    else:
        raise NotImplementedError


def adam_modifier(parser: ArgumentParser, *_):
    parser.add_argument("--b1", type=float, default=0.9, help="Adam b1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam b2")
    return parser


def adabound_modifier(parser: ArgumentParser, *_):
    parser = adam_modifier(parser)
    parser.add_argument("--final_lr", type=float, default=0.1, help="AdaBound final_lr")
    return parser


def define_optimizer(parameters, opt, net: str) -> torch.optim.Optimizer:
    """
    Return an initialized Optimizer class
    :param opt:
    :param net:
    :param parameters:
    :return:
    """
    # check whether optimizer_G or optimizer_D
    if net != "D" and net != "G":
        raise ValueError(f"net arg must be 'D' or 'G', received {net}")
    arg = "optimizer_" + net
    choice = getattr(opt, arg)

    # add optimizer kwargs
    lr = opt.d_lr if net == "D" else opt.lr
    wd = opt.d_weight_decay if net == "D" else opt.weight_decay
    kwargs = {"lr": lr, "weight_decay": wd, "betas": (opt.b1, opt.b2)}
    if choice == "AdaBound":
        kwargs["final_lr"] = opt.final_lr

    optim_class = eval(choice)
    optimizer = optim_class(parameters, **kwargs)
    return optimizer
