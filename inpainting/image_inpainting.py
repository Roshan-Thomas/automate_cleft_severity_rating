from processing import StableDiffusionProcessingImg2Img
from PIL import Image, ImageOps, ImageChops




infotext_to_setting_name_mapping = [
    ('Clip skip', 'CLIP_stop_at_last_layers',),
    ('Conditional mask weight', 'inpainting_mask_weight'),
    ('Model hash', 'sd_model_checkpoint'),
    ('ENSD', 'eta_noise_seed_delta'),
    ('Noise multiplier', 'initial_noise_multiplier'),
    ('Eta', 'eta_ancestral'),
    ('Eta DDIM', 'eta_ddim'),
    ('Discard penultimate sigma', 'always_discard_next_to_last_sigma'),
    ('UniPC variant', 'uni_pc_variant'),
    ('UniPC skip type', 'uni_pc_skip_type'),
    ('UniPC order', 'uni_pc_order'),
    ('UniPC lower order final', 'uni_pc_lower_order_final'),
]

def create_override_settings_dict(text_pairs):
    """
    Creates processing's override_settings parameters from gradio's multiselect
    """

    res = {}

    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)

        params[k] = v.strip()

    for param_name, setting_name in infotext_to_setting_name_mapping:
        value = params.get(param_name, None)

        if value is None:
            continue

        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res

def img2img(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch,
            inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, 
            inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float,
            seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int,
            resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str,
            img2img_batch_inpaint_mask_dir: str, override_settings_texts, *args):
  
    override_settings = create_override_settings_dict(override_settings_texts)

    # Inpaint mode (mode 2)

    image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
    alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
    mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
    image = image.convert("RGB")

    # Use the EXIF orientation of photos taken by smartphones (rotate images)
    if image is not None: 
        image = ImageOps.exif_transpose(image)
    
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
    )

    # ....    



