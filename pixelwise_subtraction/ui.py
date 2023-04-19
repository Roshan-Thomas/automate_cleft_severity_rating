import gradio as gr
from pixelwise_subtraction.pixelwise_subtraction import pixelwise_subtraction_gradio

def pixelwise_subtraction_ui():
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
    submit_button.click(
        fn=pixelwise_subtraction_gradio, 
        inputs=[original_image, normalized_image], 
        outputs=[difference_image, heatmap_image, severity_rating]
    )