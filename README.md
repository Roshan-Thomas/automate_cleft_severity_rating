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
