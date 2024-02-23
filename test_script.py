import argparse
import json
import os
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch
from einops import rearrange
from copy import deepcopy
import torch.nn.functional as F
from tqdm import tqdm
from utils.mvdream2diffuser import convert_from_original_mvdream_ckpt
from utils.custom_utils import drag_diffusion_update, drag_diffusion_update_gen, load_and_preprocess_images
from utils.custom_attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.mv_unet import LoRAMemoryEfficientCrossAttention
import matplotlib.pyplot as plt

from utils.mv_unet import get_camera

from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms


parser = argparse.ArgumentParser(description='User given params')

parser.add_argument("--folder_path", default='/data22/DISCOVER_summer2023/shenst2306/drag_diff2/DragDiffusion/data/lego_test', help='folder path where images, masks and camera poses are saved')
parser.add_argument("--ckpt_dir", default='/data22/DISCOVER_summer2023/shenst2306/.cache/huggingface/hub/models--MVDream--MVDream/snapshots/d14ac9d78c48c266005729f2d5633f6c265da467/sd-v2.1-base-4view.pt', help="checkpoint dir")
parser.add_argument("--config_dir", default='/data22/DISCOVER_summer2023/shenst2306/drag_diff2/MVDream/mvdream/configs/sd-v2-base.yaml', help="MVdream config dir")
parser.add_argument("--is_mask_given", default=False, help="whether masks.json is given")
parser.add_argument("--is_camera_given", default=True, help="whether camera_pose.json is given")
parser.add_argument("--is_lora", default=True, help="whether use lora finetune")
parser.add_argument("--prompt", default='a green chair', help="prompt of MVdream model")
parser.add_argument("--lora_lr", default=0.0005, help="lora learning rate")
parser.add_argument("--lora_step", default=300, type=int, help="step number of lora finetuning")

args = parser.parse_args()

# handle_points = torch.tensor([[[60.,53.]],[[125., 44.]],[[195.,45.]],[[128.,60.]]]) 
# target_points = torch.tensor([[[45.,31.]],[[125., 28.]],[[211.,34.]],[[128.,40.]]]) 
# handle_points = torch.tensor([[[390, 154]],[[227,127]],[[360, 123]],[[560, 110]]]).float() / 2
# target_points = torch.tensor([[[390, 84]],[[227, 61]],[[360, 62]],[[560, 34]]]) .float() / 2
# target_points = torch.tensor([[[390, 250]],[[227, 200]],[[360, 200]],[[560, 200]]]) .float() / 2
# handle_points = torch.tensor([[[230, 130]], [[564, 208]], [[388, 146]], [[362, 120]]]).float() / 2
# target_points = torch.tensor([[[274, 296]],[[515, 279]],[[399, 354]],[[358, 280]]]) .float() / 2

# lego:
handle_points = torch.tensor([[[630, 235], [600, 264], [617, 244], [664, 221]], [[190, 235], [135, 255], [148, 231], [214, 250]], [[148, 229], [213, 189], [204, 209], [136, 240]], [[650, 217], [653, 226], [641, 219], [576, 214]]]).float() / 2
target_points = torch.tensor([[[630, 427], [600, 388], [617, 381], [664, 363]],[[190, 427], [135, 360], [148, 379], [214, 391]],[[148, 421], [213, 332], [204, 346], [136, 353]],[[650, 409], [653, 374], [641, 356], [576, 350]]]) .float() / 2

if args.is_mask_given:
    mask_dir = os.path.join(args.folder_path, 'masks.json')
else:
    mask_dir = None

inversion_strength = 1.0
latent_lr = 0.01
n_pix_step = 80
lam = 0.1

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
source_images, imgname2idx = load_and_preprocess_images(args.folder_path, device)
print("shape of input image:", source_images.shape)
full_h, full_w = source_images.shape[-2:]
args.sup_res_h = full_h
args.sup_res_w = full_w
args.sup_res_h = int(0.5*full_h)
args.sup_res_w = int(0.5*full_w)
start_step = 0
start_layer = 8

# it has to be
lora_batch_size = 1

# processed_images is a list of tensors with shape n c h w
model = convert_from_original_mvdream_ckpt(args.ckpt_dir, args.config_dir, device)

if not args.is_camera_given:
    model.camera = get_camera(4, elevation=15, extra_view=False).to(dtype=source_images.dtype, device=device)
else:
    f = open(os.path.join(args.folder_path, 'camera_pose.json'), 'r')
    content = f.read()
    data = json.loads(content)['frames']
    infor_needed = []
    for one_pose in data:
        name = one_pose['file_path'].split('/')[-1]
        idx = imgname2idx[name]
        infor_needed.append((idx, one_pose['transform_matrix']))
    print(sorted(infor_needed))
    camera_poses = [pose for _, pose in sorted(infor_needed)]
    # print(camera_poses.shape)
    camera_poses = torch.tensor(camera_poses).reshape(len(camera_poses), -1).to(device)
    model.camera = camera_poses

def print_param_stats(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            print(f'{name}: mean={param.data.mean()}, std={param.data.std()}')

# obtain text embeddings
text_embeddings = model._encode_prompt(args.prompt, device=device, num_images_per_prompt=4, do_classifier_free_guidance=args.guidance_scale>1.)

if args.is_lora:
    modified_unet = deepcopy(model.unet)
    # initialize UNet LoRA
    for name, module in model.unet.named_modules():
        if name.endswith('attn1') or name.endswith('attn2'):
            path_components = name.split('.')
            submodule = modified_unet
            for component in path_components:
                submodule_pre = submodule
                submodule = getattr(submodule, component)
            # print("!n!")
            # print(submodule)
            q_dim = getattr(getattr(submodule, 'to_q'), 'in_features')
            context_dim = getattr(getattr(submodule, 'to_k'), 'in_features')
            heads = getattr(submodule, 'heads')
            dim_head = getattr(submodule, 'dim_head')
            # print(q_dim)
            # print(context_dim)
            lora_attn = LoRAMemoryEfficientCrossAttention(
                query_dim=q_dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head
                )

            # 替换最后一个组件
            setattr(submodule_pre, path_components[-1], lora_attn)
    # print("before:")
    # for name, module in model.unet.named_modules():
    #     print_param_stats(modified_unet, name)
    msg = modified_unet.load_state_dict(model.unet.state_dict(), strict=False)
    # print(msg)

    model.unet = modified_unet.to(device)
    del modified_unet
    
    
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
    set_seed(0)
    lora_parameters = []
    for name, param in model.unet.named_parameters():
        if 'lora' in name:
            lora_parameters.append(param)
            # print(name)
    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=args.lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.lora_step,
        num_cycles=1,
        power=1.0,
    )
    # # prepare accelerator
    # unet_lora_layers = accelerator.prepare_model(model.unet)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    # initialize text embeddings
    with torch.no_grad():

        text_embeddings = text_embeddings.repeat(lora_batch_size, 1, 1)

    # initialize latent distribution
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    noise_scheduler = model.scheduler
    for step in tqdm(range(args.lora_step), desc="training LoRA"):
        model.unet.train()
        # image_batch = []
        # for _ in range(lora_batch_size):
        #     image_transformed = image_transforms(Image.fromarray(image)).to(device, dtype=torch.float16)
        #     image_transformed = image_transformed.unsqueeze(dim=0)
        #     image_batch.append(image_transformed)

        # # repeat the image_transformed to enable multi-batch training
        # image_batch = torch.cat(image_batch, dim=0)

        model_input = latents = model.encode_image_latents(source_images, device=device, num_images_per_prompt=4)[1]
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (lora_batch_size,), device=model_input.device
        )
        timesteps = timesteps.long()
        timesteps = torch.cat([timesteps] * 4, dim=0)
        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        unet_inputs = {
                        'x': noisy_model_input,
                        'timesteps': timesteps,
                        'context': text_embeddings,
                        'num_frames': 4,
                        'camera': model.camera,
                    }
        # Predict the noise residual
        model_pred = model.unet.forward(**unet_inputs)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # if save_interval > 0 and (step + 1) % save_interval == 0:
        #     save_lora_path_intermediate = os.path.join(save_lora_path, str(step+1))
        #     if not os.path.isdir(save_lora_path_intermediate):
        #         os.mkdir(save_lora_path_intermediate)
        #     # unet = unet.to(torch.float32)
        #     # unwrap_model is used to remove all special modules added when doing distributed training
        #     # so here, there is no need to call unwrap_model
        #     # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
        #     LoraLoaderMixin.save_lora_weights(
        #         save_directory=save_lora_path_intermediate,
        #         unet_lora_layers=unet_lora_layers,
        #         text_encoder_lora_layers=None,
        #     )

print("start pipeline~")



# invert the source image
# the latent code resolution is too small, only 64*64
invert_code = model.invert(source_images,
                        args.prompt,
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


if mask_dir is not None:
    idx2name = dict(zip(imgname2idx.values(), imgname2idx.keys()))
    with open(mask_dir) as f:
        masks_dict = json.load(f)
    masks = []
    for i in range(1, 5):
        print(idx2name)
        mask = masks_dict[idx2name[i]+'.png']
        masks.append(mask)
        # Visualization
        plt.figure(figsize=(6, 6))
        plt.imshow(1 - np.array(mask), cmap='gray')
        plt.colorbar()
        plt.title("Visualization of the 2D Mask")
        plt.axis('off')  # Hide the axis

        # Save the visualization result
        file_path = "mask_visualization_{}.png".format(i)
        plt.savefig(file_path)
        
        
    mask = 1 - torch.tensor(masks)
    # print(mask)
    mask = rearrange(mask, "n h w -> n 1 h w").float().cuda()
else:
    mask = torch.ones_like(source_images[0, 0, :, :])
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()



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
if args.is_lora:
    register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_mv')
else:
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

os.makedirs("./MV_drag_output", exist_ok=True)
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