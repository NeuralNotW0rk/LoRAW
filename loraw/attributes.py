# A dict mapping model types to the location of their EMA object within the model wrapper
EMA_MODEL = {
    "diffusion_uncond": "diffusion_ema",
    "diffusion_cond": "diffusion_ema",
    "diffusion_cond_inpaint": "diffusion_ema",
    "diffusion_autoencoder": "diffae_ema",
}
