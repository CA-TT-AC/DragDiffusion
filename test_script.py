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
lora_path = ""

device = 'cuda'
prompt = 'Head of Hatsune Miku'
# prompt = ''
inversion_strength = 1.0
latent_lr = 0.01
n_pix_step = 80
lam = 0.1
args = SimpleNamespace()
args.prompt = prompt
# args.points = points
args.n_inference_step = 50
args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
args.guidance_scale = 1.1

args.unet_feature_idx = [3]

args.r_m = 1
args.r_p = 3
args.lam = lam

args.lr = latent_lr
args.n_pix_step = n_pix_step


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_images = load_and_preprocess_images(folder_path, device)
print("shape of input image:", source_images.shape)

full_h, full_w = source_images.shape[-2:]
args.sup_res_h = int(0.5*full_h)
args.sup_res_w = int(0.5*full_w)
start_step = 0
start_layer = 10


# processed_images is a list of tensors with shape n c h w
model = convert_from_original_mvdream_ckpt(ckpt_dir, config_dir, device)
print("start pipeline~")


# obtain text embeddings
text_embeddings = model._encode_prompt(prompt, device=device, num_images_per_prompt=4, do_classifier_free_guidance=True)
# invert the source image
# the latent code resolution is too small, only 64*64
invert_code = model.invert(source_images,
                        prompt,
                        text_embeddings=text_embeddings,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.n_inference_step,
                        num_actual_inference_steps=args.n_actual_inference_step)

print("invert code shape:", invert_code[0].shape)

torch.cuda.empty_cache()

init_code = invert_code
init_code_orig = deepcopy(init_code)

model.scheduler.set_timesteps(args.n_inference_step)
t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

# feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
# update according to the given supervision
init_code = init_code.float()
# text_embeddings = text_embeddings.float()
model.unet = model.unet.float()


mask = torch.ones_like(source_images[0, 0, :, :])
mask = rearrange(mask, "h w -> 1 1 h w").cuda()
handle_points = torch.tensor([[[106., 90.]]]*4) // 2
target_points = torch.tensor([[[108., 116.]]]*4) // 2
updated_init_code = drag_diffusion_update(model, init_code,
    text_embeddings, t, handle_points, target_points, mask, args)
# updated_init_code = init_code
print(updated_init_code.shape)
print("finish update!")
# updated_init_code = updated_init_code.half()
# text_embeddings = text_embeddings.half()


# hijack the attention module
# inject the reference branch to guide the generation
editor = MutualSelfAttentionControl(start_step=start_step,
                                    start_layer=start_layer,
                                    total_steps=args.n_inference_step,
                                    guidance_scale=args.guidance_scale)
if lora_path == "":
    register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    
    
scaled_text_embedding = torch.cat([text_embeddings[:4,:,:],text_embeddings[:4,:,:],text_embeddings[4:,:,:],text_embeddings[4:,:,:]], dim=0)
# inference the synthesized image
gen_image = model.decoder(
    prompt=args.prompt,
    text_embeddings=scaled_text_embedding,
    latents=torch.cat([init_code_orig, updated_init_code], dim=0),
    output_type="pil",
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.n_inference_step,
    # num_inference_steps=args.n_actual_inference_step
    )


for i, image in enumerate(gen_image):
    image.save(f"MV_drag_output/image_{i}.png")  # type: ignore

# test MVdream inference

# images = model(
#     prompt="Head of Hatsune Miku",
#     negative_prompt="painting, bad quality, flat",
#     output_type="pil",
#     guidance_scale=7.5,
#     num_inference_steps=50,
#     device=device,
# )
# # print(images.shape)
# for i, image in enumerate(images):
#     image.save(f"test_output/test_image_{i}.png")  # type: ignore

# print(model.image_encoder)