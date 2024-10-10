from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import cv2
import mediapy
from PIL import Image
import os

#custom imports
import sa_handler
import pipeline_calls
import inversion

# setup cache path for huggingface
os.environ['CACHE_DIR'] = '/mnt/users_scratch/astitva/CACHE/'
os.environ['HF_HUB_OFFLINE'] = '1' # set it to 0 to download checkpoints for the first time
os.environ['HF_HOME'] = os.environ['CACHE_DIR']
os.environ['HF_DATASETS_CACHE'] = os.environ['CACHE_DIR']
os.environ['TRANSFORMERS_CACHE']= os.environ['CACHE_DIR']

#===================================================================================================

# configuration
num_inference_steps = 50
style_from_referemce_image = True
num_images_per_prompt = 1
control_type = "canny" # "canny" or "depth"
preprocessor = True
controlnet_conditioning_scale = 0.99
seed = 999

# reference style image and prompt
ref_image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
ref_image = load_image(ref_image_path)
ref_style = "hand drawn"
ref_prompt = f"2D character face, {ref_style}."

# condition image for control
cond_image_path = "https://www.thespruce.com/thmb/iMt63n8NGCojUETr6-T8oj-5-ns=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/PAinteriors-7-cafe9c2bd6be4823b9345e591e4f367f.jpg"
cond_image = load_image(cond_image_path)

# prompts to generate
target_prompts = ['a medieval-room with a dirty bed',
                  'a modern bedroom with clean orange-color bed',
]

#save directory
save_dir = "./output/"
os.makedirs(save_dir, exist_ok=True)

#===================================================================================================

if preprocessor:
    if control_type == "canny":
        canny_image = cv2.Canny(np.asarray(cond_image).astype('uint8'), 5, 45)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        cond_image = Image.fromarray(canny_image).resize((1024, 1024), 0)
    elif control_type == "depth":
        depth_image = pipeline_calls.get_depth_map(cond_image, feature_processor, depth_estimator)
        cond_image = depth_image

# load controlnet model
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

# DDIM scheduler
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    clip_sample=False, set_alpha_to_one=False)

# controlnet SDXL pipeline
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
    scheduler=scheduler
).to("cuda")

# style parameters
shared_score_shift = np.log(2)
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


# initialize random latents
g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(999)
latents = torch.randn(1+len(target_prompts), 4, 128, 128, dtype=pipeline.unet.dtype,).to('cuda:0') 

if style_from_referemce_image:
    # latent inversion
    print("Running inversion...")
    x0 = np.array(ref_image.resize((1024, 1024)))
    zts = inversion.ddim_inversion(pipeline, x0, ref_prompt, num_inference_steps, 2)
    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
    latents[0] = zT


print("Generating...")
all_prompts = [ref_prompt] + [f"{prompt}, {ref_style}" for prompt in target_prompts]
breakpoint()
images = pipeline(all_prompts,
                latents=latents,
                image=[cond_image]*len(all_prompts),
                controlnet_conditioning_scale=0.99,
                callback_on_step_end=inversion_callback,
                num_inference_steps=num_inference_steps,
                guidance_scale=10).images

# saving images except the reference image
for idx, image in enumerate(images[1:]):
    image.save(f"{save_dir}/generated_{idx}.png")
