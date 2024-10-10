import os
# setup cache path for huggingface
os.environ['CACHE_DIR'] = '/mnt/users_scratch/astitva/CACHE/'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

print('HF_HOME',os.environ['HF_HOME'])
print('HF_DATASETS_CACHE',os.environ['HF_DATASETS_CACHE'])
print('TRANSFORMERS_CACHE',os.environ['TRANSFORMERS_CACHE'])

from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import cv2
import mediapy
from PIL import Image

import sa_handler
import pipeline_calls
import inversion

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")


scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
    scheduler=scheduler
).to("cuda")

# # inversion pipeline
# inversion_pipeline = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
#     use_safetensors=True,
#     scheduler=scheduler
# ).to("cuda")


shared_score_shift = np.log(4)
shared_score_scale = 1.0
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=True,
                                      share_layer_norm=True,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                      shared_score_shift=shared_score_shift,
                                      shared_score_scale=shared_score_scale,
                                     )
handler = sa_handler.Handler(pipeline)
handler.register(sa_args, )


ref_image = load_image("./example_image/face_2/main.png")
ref_style = "hand drawn"
ref_prompt = f"2D character face, {ref_style}."
num_inference_steps = 50
image_inversion = True
num_images_per_prompt = 1

cond_image_path = "./example_image/face_2/3.png"
cond_image = load_image(cond_image_path)
# depth
depth_image = pipeline_calls.get_depth_map(cond_image, feature_processor, depth_estimator)
#canny
canny_image = cv2.Canny(np.asarray(cond_image).astype('uint8'), 5, 45)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image).resize((1024, 1024), 0)

target_prompt = f"2D character face red tongue and white teeth, {ref_style}."

# initialize random latents
# g_cpu = torch.Generator(device='cpu')
# g_cpu.manual_seed(999)
latents = torch.randn(1+num_images_per_prompt, 4, 128, 128, dtype=pipeline.unet.dtype,).to('cuda:0') 

if image_inversion:
    # latent inversion
    print("Running inversion...")
    x0 = np.array(ref_image.resize((1024, 1024)))
    zts = inversion.ddim_inversion(pipeline, x0, ref_prompt, num_inference_steps, 2)
    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
    latents[0] = zT


print("Generating...")
# images = pipeline_calls.controlnet_call(pipeline, [ref_prompt, target_prompt],
#                                         image=canny_image,
#                                         num_inference_steps=50,
#                                         controlnet_conditioning_scale=controlnet_conditioning_scale,
#                                         num_images_per_prompt=1,
#                                         latents=latents)
images = pipeline([ref_prompt, target_prompt],
                latents=latents,
                image=canny_image,
                controlnet_conditioning_scale=0.99,
                callback_on_step_end=inversion_callback,
                num_inference_steps=num_inference_steps,
                guidance_scale=10).images


images[1].resize(cond_image.size).save(f"{cond_image_path[:-4]}_stylized.png")
