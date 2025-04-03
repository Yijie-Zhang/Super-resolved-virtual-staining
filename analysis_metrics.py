
import argparse
import os
import blobfile as bf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import csv

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import torch

from torchvision import transforms
from scipy.stats import pearsonr

import csv


def evaluate_images(output_path, target_path):
    output_img = np.array(Image.open(output_path).convert('RGB'))
    
    target_img = np.array(Image.open(target_path).convert('RGB'))

    psnr = compare_psnr(target_img, output_img, data_range=target_img.max() - target_img.min())
    ssim = 1 / 3 *(compare_ssim(np.squeeze(target_img[:,:,0]), np.squeeze(output_img[:,:,0]))+
                   compare_ssim(np.squeeze(target_img[:,:,1]), np.squeeze(output_img[:,:,1]))+
                   compare_ssim(np.squeeze(target_img[:,:,2]), np.squeeze(output_img[:,:,2])))

    return psnr, ssim


def extract_slide_fov_names(path):
    """
    Extract slide name and FOV name from the path.
    Assumes the format `.../slide_name/fov_name_xx.png`.
    """
    parts = os.path.normpath(path).split(os.sep)
    slide_name = parts[-2]
    fov_name = os.path.splitext(parts[-1])[0]
    return slide_name, fov_name

def compute_metrics_for_model(model_paths, gt_paths, model_name, output_dir):
    lpips_vgg = lpips.LPIPS(net='vgg').to('cuda')
    all_metrics = []

    for model_path, gt_path in zip(model_paths, gt_paths):
        output_img = Image.open(model_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        psnr, ssim = evaluate_images(model_path, gt_path)

        output_tensor = transforms.ToTensor()(output_img).unsqueeze(0).to('cuda')
        gt_tensor = transforms.ToTensor()(gt_img).unsqueeze(0).to('cuda')
        lpips_score = lpips_vgg(output_tensor, gt_tensor).mean().item()

        slide_name_gt, fov_name_gt = extract_slide_fov_names(gt_path)
        slide_name, fov_name = extract_slide_fov_names(model_path)

        # remove "_target" & "_gt" from the fov_name
        fov_name = fov_name.replace("_gt", "")
        fov_name_gt = fov_name_gt.replace("_gt", "")

        # assert join names are the same
        assert slide_name == slide_name_gt
        assert fov_name == fov_name_gt

        all_metrics.append([slide_name, fov_name, psnr, ssim, lpips_score])

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{model_name}_metrics.csv")

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Slide Name', 'FOV Name', 'PSNR', 'SSIM', 'LPIPS'])
        writer.writerows(all_metrics)

    print(f"Metrics saved for model: {model_name}")

if __name__ == "__main__":
    

    path_gt = glob.glob(r'I:\sr_vs_revision\outputs_1x_diff\*\*_gt.png')
    
    path_1x = glob.glob(r'I:\sr_vs_revision\outputs_1x_diff\*\*_gt.png') 
    path_1x = [path.replace('_gt.png','.png') for path in path_1x]

    path_2x = glob.glob(r'I:\sr_vs_revision\outputs_2x_diff\*\*_gt.png') 
    path_2x = [path.replace('_gt.png','.png') for path in path_2x]

    path_3x = glob.glob(r'I:\sr_vs_revision\outputs_3x_diff\*\*_gt.png') 
    path_3x = [path.replace('_gt.png','.png') for path in path_3x]

    path_4x = glob.glob(r'I:\sr_vs_revision\outputs_4x_diff\*\*_gt.png') 
    path_4x = [path.replace('_gt.png','.png') for path in path_4x]

    path_5x = glob.glob(r'I:\sr_vs_revision\outputs_5x_diff\*\*_gt.png') 
    path_5x = [path.replace('_gt.png','.png') for path in path_5x]

    path_1x_cgan = glob.glob(r'I:\sr_vs_revision\outputs_1x_cgan\*\*_gt.png') 
    path_1x_cgan = [path.replace('_gt.png','.png') for path in path_1x_cgan]

    path_2x_cgan = glob.glob(r'I:\sr_vs_revision\outputs_2x_cgan\*\*_gt.png') 
    path_2x_cgan = [path.replace('_gt.png','.png') for path in path_2x_cgan]
    
    path_3x_cgan = glob.glob(r'I:\sr_vs_revision\outputs_3x_cgan\*\*_gt.png') 
    path_3x_cgan = [path.replace('_gt.png','.png') for path in path_3x_cgan]

    path_4x_cgan = glob.glob(r'I:\sr_vs_revision\outputs_4x_cgan\*\*_gt.png') 
    path_4x_cgan = [path.replace('_gt.png','.png') for path in path_4x_cgan]

    path_5x_cgan = glob.glob(r'I:\sr_vs_revision\outputs_5x_cgan\*\*_gt.png') 
    path_5x_cgan = [path.replace('_gt.png','.png') for path in path_5x_cgan]


    
    path_gt = sorted(path_gt)
    model_paths = {
        "diffusion_1x": sorted(path_1x),
        "diffusion_2x": sorted(path_2x),
        "diffusion_3x": sorted(path_3x),
        "diffusion_4x": sorted(path_4x),
        "diffusion_5x": sorted(path_5x),
        "cgan_1x": sorted(path_1x_cgan),
        "cgan_2x": sorted(path_2x_cgan),
        "cgan_3x": sorted(path_3x_cgan),
        "cgan_4x": sorted(path_4x_cgan),
        "cgan_5x": sorted(path_5x_cgan),
    }

    output_dir = r"I:\BBDM_model_tree_revision\analysis_revision_results\all_lung_metrics"
    # Process each model
    for model_name, paths in model_paths.items():
        if len(paths) != len(path_gt):
            print(f"Warning: Mismatch in GT and model paths for {model_name}")
            continue
        compute_metrics_for_model(paths, path_gt, model_name, output_dir)
