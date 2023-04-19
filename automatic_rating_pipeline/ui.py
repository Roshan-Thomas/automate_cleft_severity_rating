import gradio as gr
from automatic_rating_pipeline.pipeline import automatic_rating_pipeline

def greet(img):
    severity_rating = "3.736"
    return (f"Severity Rating: {severity_rating}")


def automatic_rating_ui():
    with gr.Row():
        with gr.Column():
            input_image = gr.Image()
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            severity_rating = gr.Label(label="Severity Rating")
            classifier_image = gr.Image(label="Image with Mask Applied")
            image_inpainted_image = gr.Image(label="Normalized Image")

    submit_button.click(
        fn=automatic_rating_pipeline,
        inputs=input_image,
        outputs=[severity_rating, classifier_image, image_inpainted_image]
    )

    gr.Examples(
        examples=["examples/1.png", "examples/3.png", "examples/4.png", "examples/8.png", "examples/12.png"],
        inputs=input_image,
        outputs=[severity_rating, classifier_image, image_inpainted_image],
        fn=automatic_rating_pipeline,
    )