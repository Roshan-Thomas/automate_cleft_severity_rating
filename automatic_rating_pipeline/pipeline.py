from pixelwise_subtraction.pixelwise_subtraction import pixelwise_subtraction_gradio
from image_inpainting.ui import image_inpainting
from cnn_classifier.ui import detect_nasolabial_area

def automatic_rating_pipeline(input_image):
    # classifier
    masked_nasolabial_area_image, _ = detect_nasolabial_area(input_image)

    print("[INFO] Identified Nasolabial Area.")

    # stable diffusion
    normalized_image = image_inpainting(masked_nasolabial_area_image,
                                        sampling_method="Euler a",
                                        sampling_steps=20,
                                        width=512,
                                        height=512,
                                        cfg_scale=7,
                                        denoising_strength=0.75,
                                        seed=92548865)

    print("[INFO] Completed Image Inpainting.")

    # pixel wise subtraction
    _, _, severity_rating = pixelwise_subtraction_gradio(input_image, normalized_image)

    print("[INFO] Generated Severity Score")

    return severity_rating