import kagglehub
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import transforms as T, utils
import math

# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", path)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 2), # [64, 64, 128, 128]
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,    # T --> time step
    sampling_timesteps=500, # ddim sampling을 활용
    beta_schedule = 'linear',
)

trainer = Trainer(
    diffusion,
    os.path.join(path, "img_align_celeba/img_align_celeba"),
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 100000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    num_fid_samples = 32 * 10,
    save_best_and_latest_only=True # 가장 best fid를 가지는 model을 milestone으로 저장하겠다.
)

# trainer.train()

trainer.load(100)

num_samples = 64
all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), [32, 32]))
all_images = torch.cat(all_images_list, dim = 0)

utils.save_image(all_images, "./results/sample-100-test.png", nrow = int(math.sqrt(num_samples)))