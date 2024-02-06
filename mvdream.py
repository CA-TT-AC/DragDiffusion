import os
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch
from einops import rearrange
from copy import deepcopy

from utils.mvdream2diffuser import convert_from_original_mvdream_ckpt
from utils.custom_utils import drag_diffusion_update, drag_diffusion_update_gen
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl

def preprocess_image(image, device, dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image

def load_and_preprocess_images(folder_path, device):
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_np = np.array(image)
            processed_image = preprocess_image(image_np, device)
            processed_images.append(processed_image)
    processed_images = torch.concat(processed_images, dim=0)
    return processed_images

# Example usage
folder_path = '/data22/DISCOVER_summer2023/shenst2306/drag_diff2/DragDiffusion/data/mvdream_output'
ckpt_dir = '/data22/DISCOVER_summer2023/shenst2306/.cache/huggingface/hub/models--MVDream--MVDream/snapshots/d14ac9d78c48c266005729f2d5633f6c265da467/sd-v2.1-base-4view.pt'
config_dir = '/data22/DISCOVER_summer2023/shenst2306/drag_diff2/MVDream/mvdream/configs/sd-v2-base.yaml'
device = 'cuda'
prompt = 'a green chair'
# processed_images is a list of tensors with shape n c h w
model = convert_from_original_mvdream_ckpt(ckpt_dir, config_dir, device)
print("start~")
images = model(
    prompt=prompt,
    negative_prompt="painting, bad quality, flat",
    output_type="pil",
    guidance_scale=7.5,
    num_inference_steps=50,
    device=device,
)
# print(images.shape)
for i, image in enumerate(images):
    image.save(f"test_output/test_image_{i}.png")  # type: ignore
