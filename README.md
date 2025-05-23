
## Codes for "Super-resolved Virtual Staining of Label-free Tissue Using Diffusion Models" [[paper]](https://arxiv.org/pdf/2410.20073)

## Installation

Set up the environment using the `conda` environment file:

```bash
conda env create -f environment.yml
```

## Train the model

```bash
python train.py
``` 

## Download Test Samples and Pretrained Model

Download the test samples and pretrained model from the following [Google Drive link](https://drive.google.com/drive/folders/1R9V5UtmlYHpGqQ_gjv02DH5QInz2kJ8k?usp=drive_link).
.

After downloading, place the "test_samples" and "models" folders under the same directory as the project.


## Inference on Test Samples

To run inference using different sampling strategies, execute the following commands:

```bash
python test_diffusion_reverse_skip_sample_strategy.py
python test_diffusion_reverse_mean_sample_strategy.py
``` 

## Evaluate results
To calculate the metrics, execute the following commands:

```bash
python analysis_metrics.py
``` 
