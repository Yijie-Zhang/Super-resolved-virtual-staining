"""
Train a super-resolution virtual staining model. 
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from datetime import datetime

import numpy as np
import torch as torch
import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_paired_npy_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.unet import CompressCNN_upsample as CompressCNN

channel_in = 4
channel_out = 3

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()


    # Create diffusion model
    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()),
        in_channels = 3, out_channels = 3
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    
    # Create shallow CNN model
    model_compressor = CompressCNN(channel_in, channel_out, dims=2)
    model_compressor.to(dist_util.dev())

    # Create data loader
    logger.log("creating data loader...")
    data = load_pair_superres_data(
        args.hr_data_dir,
        args.lr_data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )

    # Train model
    logger.log("training...")
    TrainLoop(
        model=model,
        model_compressor=model_compressor,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        resume_checkpoint=args.resume_checkpoint,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_pair_superres_data(hr_data_dir, lr_data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_paired_npy_data(
        input_dir=lr_data_dir,
        target_dir=hr_data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, small_batch, model_kwargs in data:
        yield large_batch, small_batch, model_kwargs


def create_argparser():
    defaults = dict(
        lr_data_dir="./Training/*/input", # training input data directory
        hr_data_dir="./Training/*/target", # training target data directory
        schedule_sampler="uniform",
        lr=1e-4, # learning rate
        weight_decay=0.0, # weight decay
        lr_anneal_steps=0, # learning rate anneal steps
        batch_size=8, # batch size
        microbatch=-1, # microbatch size
        ema_rate="0.9999", 
        log_interval=10,
        save_interval=2000,
        val_interval=1000,
        resume_checkpoint="",
        model_dir="models",
        log_dir="log",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    defaults['model_dir'] = os.path.join(defaults['model_dir'], 'BBDM-%s'%datetime_str)
    defaults['log_dir'] = os.path.join(defaults['log_dir'], 'BBDM-%s'%datetime_str)
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
