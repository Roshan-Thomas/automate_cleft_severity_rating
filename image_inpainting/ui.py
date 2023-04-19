import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
import os
import imutils
from PIL import Image
import cv2
import numpy as np


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def image_inpainting(input_image, mask_image, cfg_scale, seed):
    device = "cuda"
    if torch.cuda.is_available():
        generator = torch.Generator('cuda').manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    color_converted_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(color_converted_input)

    mask_image = imutils.resize(mask_image, width=512)
    mask_image = Image.fromarray(mask_image)

    model_path = "runwayml/stable-diffusion-inpainting"
    # model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    output_image = pipe(
        prompt=" ",
        image=input_image,
        mask_image=mask_image,
        guidance_scale=cfg_scale,
        generator=generator,
    ).images

    return (output_image[0])

def image_inpainting_gradio(dict, width, height, cfg_scale, seed):
    device = "cuda"

    if torch.cuda.is_available():
        generator = torch.Generator('cuda').manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))

    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    input_image = dict['image'].convert("RGB").resize((512, 512))
    mask_image = dict['mask'].convert("RGB").resize((512, 512))

    output_image = pipe(
        prompt=" ",
        image=input_image,
        mask_image=mask_image,
        guidance_scale=cfg_scale,
        generator=generator,
        height=height,
        width=width,
    ).images

    output_text = "Completed Image Inpainting Process!"

    return (output_image[0], output_text)


def image_inpainting_ui():
    with gr.Row():
        with gr.Column():
            submit_button = gr.Button("Generate", variant="primary")
            input_image = gr.Image(label="Image for inpainting with Mask", show_label=False, source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA")

            # with gr.Row():
            #     sampling_method_choices = ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++2S a", "DPM++ 2M", "DPM++ SDE", "DPM fast", "DPM adaptive", "LMS Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM"]
            #     sampling_method = gr.Dropdown(label="Sampling Method", choices=sampling_method_choices, interactive=True, value="Euler a")
            #     sampling_steps = gr.Slider(label="Sampling Steps", minimum=0, maximum=150, step=1, value=20, interactive=True)
            #     num_samples = gr.Number(label="Number of Samples", minimum=1, maximum=64, value=1, interactive=True)


            with gr.Row():
                width = gr.Slider(label="Width", minimum=0, maximum=2048, step=10, value=512, interactive=False)
                heigth = gr.Slider(label="Height", minimum=0, maximum=2048, step=10, value=512, interactive=False)
                
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.5, value=7, interactive=True)
            # denoising_strength = gr.Slider(label="Denoising Strength", minimum=0, maximum=1, step=0.01, value=0.75, interactive=True)


            seed = gr.Number(label="Seed", value=1551597449, interactive=True)

        with gr.Column():
            output_image = gr.Image()
            output_text = gr.Textbox(label="Output")


    submit_button.click(
        fn=image_inpainting_gradio,
        inputs=[input_image, width, heigth, cfg_scale, seed],
        outputs=[output_image, output_text],
    )
