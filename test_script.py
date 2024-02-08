import json
import os
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch
from einops import rearrange
from copy import deepcopy
import torch.nn.functional as F
from utils.mvdream2diffuser import convert_from_original_mvdream_ckpt
from utils.custom_utils import drag_diffusion_update, drag_diffusion_update_gen
from utils.custom_attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl

from utils.mv_unet import get_camera
def preprocess_image(image, device, dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image

def load_and_preprocess_images(folder_path, device):
    processed_images = []
    filenames = os.listdir(folder_path)
    filenames.sort()
    imgname2idx = {}
    for i, filename in enumerate(filenames):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            imgname2idx[filename.split('.')[0]] = i
            print(filename+" loaded!")
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image_np = np.array(image)
            processed_image = preprocess_image(image_np, device)
            processed_images.append(processed_image)
    processed_images = torch.concat(processed_images, dim=0)
    processed_images = processed_images[:, :3, :, :]
    return processed_images, imgname2idx

# Example usage
folder_path = '/data22/DISCOVER_summer2023/shenst2306/drag_diff2/DragDiffusion/data/mvdream_output2'
ckpt_dir = '/data22/DISCOVER_summer2023/shenst2306/.cache/huggingface/hub/models--MVDream--MVDream/snapshots/d14ac9d78c48c266005729f2d5633f6c265da467/sd-v2.1-base-4view.pt'
config_dir = '/data22/DISCOVER_summer2023/shenst2306/drag_diff2/MVDream/mvdream/configs/sd-v2-base.yaml'
lora_path = ""

device = 'cuda'
# prompt = 'Head of Hatsune Miku'
prompt = 'a green chair'
is_camera_given = False


inversion_strength = 1.0
latent_lr = 0.01
n_pix_step = 80
lam = 0.1
args = SimpleNamespace()
args.prompt = prompt
# args.points = points
args.n_inference_step = 50
args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
args.guidance_scale = 1

args.unet_feature_idx = [3]

args.r_m = 1
args.r_p = 3
args.lam = lam

args.lr = latent_lr
args.n_pix_step = n_pix_step


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_images, imgname2idx = load_and_preprocess_images(folder_path, device)
print("shape of input image:", source_images.shape)

full_h, full_w = source_images.shape[-2:]
args.sup_res_h = full_h
args.sup_res_w = full_w
# args.sup_res_h = int(0.5*full_h)
# args.sup_res_w = int(0.5*full_w)
start_step = 0
start_layer = 5


# processed_images is a list of tensors with shape n c h w
model = convert_from_original_mvdream_ckpt(ckpt_dir, config_dir, device)
print("start pipeline~")


# obtain text embeddings
text_embeddings = model._encode_prompt(prompt, device=device, num_images_per_prompt=4, do_classifier_free_guidance=args.guidance_scale>1.)

if not is_camera_given:
    model.camera = get_camera(4, elevation=15, extra_view=False).to(dtype=source_images.dtype, device=device)
else:
    f = open(os.path.join(folder_path, 'camera_pose.json'), 'r')
    content = f.read()
    data = json.loads(content)['frames']
    infor_needed = []
    for one_pose in data:
        name = one_pose['file_path'].split('/')[-1]
        idx = imgname2idx[name]
        infor_needed.append((idx, one_pose['transform_matrix']))
    print(infor_needed)
    camera_poses = [pose for _, pose in sorted(infor_needed)]
    # print(camera_poses.shape)
    camera_poses = torch.tensor(camera_poses).reshape(len(camera_poses), -1).to(device)
    model.camera = camera_poses
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
print("camera pose:", model.camera)

mask = torch.ones_like(source_images[0, 0, :, :])
mask = rearrange(mask, "h w -> 1 1 h w").cuda()
handle_points = torch.tensor([[[60.,53.]],[[125., 44.]],[[195.,45.]],[[128.,60.]]]) 
target_points = torch.tensor([[[45.,31.]],[[125., 28.]],[[211.,34.]],[[128.,40.]]]) 
# handle_points = torch.tensor([[[390, 154]],[[227,127]],[[360, 123]],[[560, 110]]]).float() / 2
# target_points = torch.tensor([[[390, 94]],[[227, 71]],[[360, 72]],[[560, 44]]]) .float() / 2
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
    register_attention_editor_diffusers(model, editor, attn_processor='attn_mv')
    
    
scaled_text_embedding = torch.cat([text_embeddings[:4,:,:],text_embeddings[:4,:,:],text_embeddings[4:,:,:],text_embeddings[4:,:,:]], dim=0)
print("scaled_text_embedding:", scaled_text_embedding.shape)
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