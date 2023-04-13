import gradio as gr

def image_inpainting(input_image, sampling_method, sampling_steps, width, height, cfg_scale, denoising_strength, seed):
    

    return output_image


def image_inpainting_ui():
    with gr.Row():
        with gr.Column():
            submit_button = gr.Button("Generate", variant="primary")
            input_image = gr.Image(label="Image for inpainting with Mask", show_label=False, source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA")

            with gr.Row():
                sampling_method_choices = ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++2S a", "DPM++ 2M", "DPM++ SDE", "DPM fast", "DPM adaptive", "LMS Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM"]
                sampling_method = gr.Dropdown(label="Sampling Method", choices=sampling_method_choices, interactive=True, value="Euler a")
                sampling_steps = gr.Slider(label="Sampling Steps", minimum=0, maximum=150, step=1, value=20, interactive=True)


            with gr.Row():
                width = gr.Slider(label="Width", minimum=0, maximum=2048, step=10, value=150, interactive=True)
                heigth = gr.Slider(label="Height", minimum=0, maximum=2048, step=10, value=150, interactive=True)
                
            cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.5, value=7, interactive=True)
            denoising_strength = gr.Slider(label="Denoising Strength", minimum=0, maximum=1, step=0.01, value=0.75, interactive=True)

            seed = gr.Textbox(label="Seed", value=-1, max_lines=1, interactive=True)
                

        with gr.Column():
            output_image = gr.Image()



    
    submit_button.click(
        fn=image_inpainting,
        inputs=[input_image, sampling_method, sampling_steps, width, heigth, cfg_scale, denoising_strength, seed],
        outputs=output_image
    )


