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


scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True,
    scheduler=scheduler
).to("cuda")


src_style = "medieval painting"
src_prompt = f'Man laying in a bed, {src_style}.'
image_path = './example_image/medieval-bed.jpeg'

num_inference_steps = 50
x0 = np.array(load_image(image_path).resize((1024, 1024)))
zts = inversion.ddim_inversion(pipeline, x0, src_prompt, num_inference_steps, 2)


prompts = [
    src_prompt,
    "a man sitting on a chair",
    "a man playing video game",
    "a man eating a burger",
]

# some parameters you can adjust to control fidelity to reference
shared_score_shift = np.log(2)  # higher value induces higher fidelity, set 0 for no shift
shared_score_scale = 1.0  # higher value induces higher, set 1 for no rescale

# for very famouse images consider supressing attention to refference, here is a configuration example:
# shared_score_shift = np.log(1)
# shared_score_scale = 0.5

for i in range(1, len(prompts)):
    prompts[i] = f'{prompts[i]}, {src_style}.'

handler = sa_handler.Handler(pipeline)
sa_args = sa_handler.StyleAlignedArgs(
    share_group_norm=True, share_layer_norm=True, share_attention=True,
    adain_queries=True, adain_keys=True, adain_values=False,
    shared_score_shift=shared_score_shift, shared_score_scale=shared_score_scale,)
handler.register(sa_args)

zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(999)

latents = torch.randn(len(prompts), 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.unet.dtype,).to('cuda:0')
latents[0] = zT

images_a = pipeline(prompts, latents=latents,
                    callback_on_step_end=inversion_callback,
                    num_inference_steps=num_inference_steps, guidance_scale=10.0).images

handler.remove()

breakpoint()
images_a[1].save("./example_image/generated.png")
