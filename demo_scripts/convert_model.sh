python scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path /mnt/petrelfs/liuwenran/repos/diffusers/resources/inpainting_ckpts/DreamShaper_5_beta2_BakedVae-inpainting.inpainting.safetensors \
    --original_config_file resources/inpainting_ckpts/v1-inference.yaml \
    --dump_path resources/inpainting_ckpts/DreamShaper_5 \
    --from_safetensors