import gradio as gr
from pixelwise_subtraction.ui import pixelwise_subtraction_ui
from image_inpainting.ui import image_inpainting_ui
from diffusers import StableDiffusionInpaintPipeline
import torch


def greet(img):
    severity_rating = "3.736"
    return(f"Severity Rating: {severity_rating}")

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;">Automatic Cleft Severity Rating</h1>
        <h3 style="text-align: center;">Roshan Thomas, Fathima Hakeem, Moussa Judia, Farah Shabbir</h3>
        """
    )

    with gr.Tab("Automatic Severity Rating"):
        image = gr.Image()
        output = gr.Label(label="Severity Rating")

        submit_button = gr.Button("Submit", variant="primary")
        submit_button.click(fn=greet, inputs=image, outputs=output)

        gr.Examples(
            examples=["./examples/1.png", "./examples/3.png", "./examples/4.png", "./examples/8.png", "./examples/12.png"],
            inputs=image,
            outputs=output,
            fn=greet,
        )

    with gr.Tab("Image Inpainting"):
        image_inpainting_ui()

    with gr.Tab("Pixel-wise Subtraction"):
        pixelwise_subtraction_ui()

    
demo.launch(debug=True)



