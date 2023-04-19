import gradio as gr

from automatic_rating_pipeline.ui import automatic_rating_ui
from pixelwise_subtraction.ui import pixelwise_subtraction_ui
from image_inpainting.ui import image_inpainting_ui
from cnn_classifier.ui import classifier_ui


with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;">Automatic Cleft Severity Rating</h1>
        <h3 style="text-align: center;">Roshan Thomas, Fathima Hakeem, Moussa Judia, Farah Shabbir</h3>
        """
    )

    with gr.Tab("Automatic Severity Rating"):
        automatic_rating_ui()

    with gr.Tab("CNN Classifer"):
        classifier_ui()

    with gr.Tab("Image Inpainting"):
        image_inpainting_ui()

    with gr.Tab("Pixel-wise Subtraction"):
        pixelwise_subtraction_ui()

    
# demo.launch(debug=True) # on linux systems

demo.launch(debug=True, server_port=8080)



