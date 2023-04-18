# Demo Day - Web Interface 

Tasks to be done

1. Upload Image to web interface
2. Send image to backend server
    * Run classifier to find cleft lip area
    * Run image inpainting - Stable Diffusion
    * Run pixel wise subtraction, lpips, ssim - generate heatmaps
    * Generate severity score
    * Send to front-end
3. Show severity score to user
4. Another tab for the user to draw their own mask on the cleft lip image, for finer control (this part is optional, but would be nice to have)

## Softwares Used
- Developed in Gradio (Python package)
- For Stable Diffusion, use one of the  [implementations here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services) to build it on our app. 


# Instructions for Development
1. Use the `dev` branch for all development purposes
    - The `main` branch will be used only for pushing major updates
2. Use a seperate branch for each module of the web interface such as `image-inpainting`, `classifier`, `pixel-wise-subtraction` etc, and then once its done, push it to the `dev` branch.
3. Once development has been finished, we push it to the `main` branch.

# Instructions for Running Project

1. Create a `conda` environment `gradio_webui_seniordesign` (or any other name, but remeber what name you called it)
    ```
    conda create --name gradio_webui_seniordesign
    ```

2. Activate conda environment
    ```
    conda activate gradio_webui_seniordesign
    ```

3. Install all the required dependencies
    ```
    pip install -r requirements.txt 
    ```

4. Start the gradio app
    ```
    gradio app.py
    ```

5. Use a local browser to view the live website. The url is http://127.0.0.1:7860/?__theme=dark



