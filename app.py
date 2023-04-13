import gradio as gr
from pixelwise_subtraction import pixelwise_subtraction_gradio


def greet(img):
    severity_rating = "5.3"
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
        image = gr.Image()
        output = gr.Image(label="Inpainted Image")

        submit_button = gr.Button("Submit", variant="primary")

        gr.Examples(
            examples=["./examples/1.png", "./examples/3.png", "./examples/4.png", "./examples/8.png", "./examples/12.png"],
            inputs=image,
            outputs=output,
            fn=greet,
        )

    with gr.Tab("Pixel-wise Subtraction"):
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=1):
                # Inputs
                original_image = gr.Image(label="Ground Truth")
                normalized_image = gr.Image(label="Normalized Image")

            with gr.Column(scale=2):
                # Outputs
                severity_rating = gr.Label(label="Severity Rating")
                difference_image = gr.Image(label="Difference Map").style(height=300)
                heatmap_image = gr.Image(label="Heatmap").style(height=300)

    
        submit_button = gr.Button("Submit", variant="primary")
        submit_button.click(fn=pixelwise_subtraction_gradio, inputs=[original_image, normalized_image], outputs=[difference_image, heatmap_image, severity_rating])
        
demo.launch()



