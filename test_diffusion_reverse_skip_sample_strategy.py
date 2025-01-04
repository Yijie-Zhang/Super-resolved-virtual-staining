
import argparse
import os, glob, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.ion()

from improved_diffusion import dist_util, logger
from improved_diffusion.unet import CompressCNN_upsample as CompressCNN
from improved_diffusion.image_datasets import load_paired_mat_data_test
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def cycle_data(data):
    while True:
        for item in data:
            yield item

def min_max_norm(img, vmin, vmax):
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)

def convert2RGB(img):
    img = ((img +1) * 127.5)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)



def main(args=None):
    if args is None:
        args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    
    #  Initialize & load diffusion model
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys(),),
        in_channels=3, out_channels=3
    )
    model_path = args.model_path.replace('_compress.pt','.pt')
    model_path_compress = args.model_path
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    # Initialize & load shallow convolution neural network model 
    model_compressor = CompressCNN(args.cond_channels, args.out_channels, dims=2)
    model_compressor.load_state_dict(
        dist_util.load_state_dict(model_path_compress, map_location="cpu")
    )
    model_compressor.to(dist_util.dev())
    model_compressor.eval()

    logger.log("loaded model %s" % args.model_path)

    logger.log("loading data...")
    data = load_pair_superres_data(args.hr_data_dir, args.lr_data_dir, args.batch_size, args.large_size, args.small_size, args.class_cond)
    
    for iter in range(0, 1):
        # set random seeds
        random.seed(iter)
        np.random.seed(iter)
        th.manual_seed(iter)
        save_dir = os.path.join(args.save_dir, 'testrun_'+str(iter))
        os.makedirs(save_dir, exist_ok=True)
        logger.log("Iteration %d: creating samples..." % iter)
        all_af, all_inp = [], []
        all_outputs, all_folder, all_files, all_targets = [], [], [], []
        n_sample = 0
        while len(all_outputs) * args.batch_size < args.num_samples:
            batch, cond, gt, model_kwargs, filename, folder = next(data)
            all_targets.append(batch.permute(0, 2, 3, 1).cpu().numpy())

            cond = cond.to(dist_util.dev())
            all_af.append(cond.permute(0, 2, 3, 1).cpu().numpy())
            cond = model_compressor(cond)
            all_inp.append(cond.permute(0, 2, 3, 1).detach().cpu().numpy())

            all_files.append(filename)
            all_folder.append(folder)
            
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            
            sample = diffusion.p_sample_mean_skip(
                model,
                cond,
                n_avg_steps=args.avg_steps,
                shape=(args.batch_size, 3, args.large_size, args.large_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample)
            for sample in all_samples:
                all_outputs.append(sample.cpu().numpy())
                n_sample += sample.shape[0]
            logger.log(f"created {len(all_outputs) * args.batch_size} samples")

        af = np.concatenate(all_af, axis=0)
        inp = np.concatenate(all_inp, axis=0)
        tag = np.concatenate(all_targets, axis=0)
        out = np.concatenate(all_outputs, axis=0)
        files = np.concatenate(all_files, axis=0)
        folders = np.concatenate(all_folder, axis=0)
        out = out[: args.num_samples]
        files = files[: args.num_samples]
        folders = folders[: args.num_samples]
        tag = tag[: args.num_samples]

        af = af[: args.num_samples]
        inp = inp[: args.num_samples]

        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in out.shape])
            out_path = os.path.join(save_dir, f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, out)
            for i in range(args.num_samples):

                inp_img = min_max_norm(inp[i],-1,1)
                out_img = convert2RGB(out[i])
                tag_img = convert2RGB(tag[i])

                save_folder = os.path.join(args.save_dir, folders[i])
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                plt.imsave(os.path.join(save_dir, files[i].replace('.mat','_inp.png')), inp_img)
                plt.imsave(os.path.join(save_dir, files[i].replace('.mat','.png')), out_img)
                plt.imsave(os.path.join(save_dir, files[i].replace('.mat','_target.png')), tag_img)


        dist.barrier()
        logger.log("sampling complete")


def load_pair_superres_data(hr_data_dir, lr_data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_paired_mat_data_test(
        input_dir=lr_data_dir,
        target_dir=hr_data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        deterministic=True,
    )
    for large_batch, small_batch, model_kwargs, files, folders in data:
        yield large_batch, small_batch, model_kwargs, files, folders


def create_argparser(avg_steps=0):
    defaults = dict(
        lr_data_dir="./test_samples/input",
        hr_data_dir="./test_samples/target",
        schedule_sampler="uniform",
        cond_channels = 4,
        out_channels = 3,
        num_samples = 1,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",
        clip_denoised=False,
        log_interval=10,
        save_interval=2000,
        val_interval=1000,
        diffusion_steps=1000,
        timestep_respacing=[1000],
        avg_steps=avg_steps,
        model_dir="models",
        model_name="BBDM-20240425-1500",
        ckpt="ema_0.9999_220000_compress.pt",
        log_dir="log",
        save_dir="outputs_5x_skip",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    if defaults['ckpt'] is not None:
        defaults['save_dir'] = os.path.join(defaults['save_dir'], defaults['model_name'], str(defaults['timestep_respacing'])+'_'+str(defaults['avg_steps'])+'_skip')
        defaults['model_path'] = os.path.join(defaults['model_dir'], defaults['model_name'], defaults['ckpt'])
    dicts = sr_model_and_diffusion_defaults()
    dicts.update(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, dicts)
    return parser

if __name__ == "__main__":
    # choose exit time step
    exit_step_list = [50] 
    for exit_step in exit_step_list:
        args = create_argparser(exit_step).parse_args()
        main(args)
