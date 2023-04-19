import gradio as gr
from automatic_rating_pipeline.pipeline import automatic_rating_pipeline

def greet(img):
    severity_rating = "3.736"
    return (f"Severity Rating: {severity_rating}")


def automatic_rating_ui():
    image = gr.Image()
    severity_rating = gr.Label(label="Severity Rating")

    submit_button = gr.Button("Submit", variant="primary")
    submit_button.click(
        fn=automatic_rating_pipeline,
        inputs=image,
        outputs=severity_rating
    )

    gr.Examples(
        examples=["examples/1.png", "examples/3.png", "examples/4.png", "examples/8.png", "examples/12.png"],
        inputs=image,
        outputs=severity_rating,
        fn=automatic_rating_pipeline,
    )