# Gradient Inversion Quick Start

## Setup

```bash
source env/bin/activate
pip install torch torchvision lpips pyyaml datasets matplotlib six
```

## Dataset Setup

```bash
# Download FFHQ images + metadata
python setup_ffhq.py --sample_size 100
```

## Basic Attack (Traditional)

```bash
python rec_mult.py --dataset FFHQ --data_path ./ffhq_dataset --model ResNet18 \
  --num_images 1 --target_id 25 --max_iterations 2000 --restarts 4 \
  --unsigned --save_image --lr 3e-2 --tv 1e-6 --bn_stat 0
```

**Expected: ~16-17 dB PSNR**

## StyleGAN2 Attack (Advanced)

1. Download `Gs.pth` StyleGAN2 model → `inversefed/genmodels/stylegan2/Gs.pth`
2. Run:

```bash
python rec_mult.py --dataset FFHQ --data_path ./ffhq_dataset --model ResNet18 \
  --num_images 1 --target_id 30 --max_iterations 1000 --restarts 2 \
  --unsigned --save_image --lr 3e-2 --tv 1e-6 --bn_stat 0 \
  --generative_model stylegan2 --gen_dataset FFHQ
```

**Expected: ~24+ dB PSNR**

## CIFAR-10 Test

```bash
python rec_mult.py --dataset CIFAR10 --model ResNet18 --num_images 1 \
  --target_id 0 --max_iterations 100 --restarts 1 --unsigned --save_image
```

## Results

- Images saved in `results/[hash]/`
- Metrics in `tables/[hash]/`
- Higher PSNR = better reconstruction

scp /Users/tanguyvans/Desktop/umons/attacks/grad.tar.gz maxglo:tanguy/
