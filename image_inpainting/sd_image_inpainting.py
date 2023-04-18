
import torch
from diffusers import StableDiffusionInpaintPipeline



def image_inpainting(input_image, sampling_method, sampling_steps, width, height, cfg_scale, denoising_strength, seed, mask_image):
    width, height, _ = input_image.shape()

    device = 'cuda'
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    
    output_image = pipe(
        prompt="",
        image=input_image,
        mask_image=mask_image,
        guidance_scale=cfg_scale,
        generator=seed,
    ).output_image

    



    return output_image