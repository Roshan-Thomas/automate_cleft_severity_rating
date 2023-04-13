from typing import Any
from PIL import Image, ImageOps, ImageChops, ImageFilter
import numpy as np


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images: list = None, resize_mode: int = 0, denoising_strength: float = 0.75, image_cfg_scale: float = None,
                 mask: Any = None, mask_blur: int = 4, inpainting_fill: int = 0, inpainting_full_res: bool = True, inpaint_full_res_padding: int = 0,
                 inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, **kwargs):
        
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.image_cfg_scale: float = image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None
        self.init_latent = None
        self.image_mask = mask
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.initial_noise_multiplier = opts.initial_noise_multiplier if initial_noise_multiplier is None else initial_noise_multiplier
        self.mask = None
        self.nmask = None
        self.image_conditioning = None

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            image_mask = image_mask.convert('L')

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)
            
            if self.mask_blur > 0:
                image_mask = image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)

            else: 
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.unit8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None

        # ....