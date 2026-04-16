import torch
import os
import argparse
import toml
from diffsynth import save_video
import requests
import time
import random
import numpy as np
import pandas as pd
from dfloat.run import DFloatDiffSynthModelFP8, get_dfloat_model_name_or_path

META_URL = (
    "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json"
)


def generate_with_intermediate_images(
    pipe, prompt, seed, num_inference_steps, save_dir, prefix="step"
):
    """Run diffusion and save decoded image at each timestep."""
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)

    pipe.scheduler.set_timesteps(num_inference_steps)

    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "cfg_scale": 1.0,
        "embedded_guidance": 3.5,
        "t5_sequence_length": 512,
        "input_image": None,
        "denoising_strength": 1.0,
        "height": 1024,
        "width": 1024,
        "seed": seed,
        "rand_device": "cpu",
        "sigma_shift": None,
        "num_inference_steps": num_inference_steps,
        "multidiffusion_prompts": (),
        "multidiffusion_masks": (),
        "multidiffusion_scales": (),
        "kontext_images": None,
        "controlnet_inputs": None,
        "ipadapter_images": None,
        "ipadapter_scale": 1.0,
        "eligen_entity_prompts": None,
        "eligen_entity_masks": None,
        "eligen_enable_on_negative": False,
        "eligen_enable_inpaint": False,
        "infinityou_id_image": None,
        "infinityou_guidance": 1.0,
        "flex_inpaint_image": None,
        "flex_inpaint_mask": None,
        "flex_control_image": None,
        "flex_control_strength": 0.5,
        "flex_control_stop": 0.5,
        "value_controller_inputs": None,
        "step1x_reference_image": None,
        "nexus_gen_reference_image": None,
        "lora_encoder_inputs": None,
        "lora_encoder_scale": 1.0,
        "tea_cache_l1_thresh": None,
        "tiled": False,
        "tile_size": 128,
        "tile_stride": 64,
        "progress_bar_cmd": tqdm,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    # Denoise
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(
        tqdm(pipe.scheduler.timesteps, desc="Denoising")
    ):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        noise_pred = pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
            progress_id=progress_id,
        )

        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"]
        )

        # Decode and save intermediate image
        pipe.load_models_to_device(["vae_decoder"])
        image = pipe.vae_decoder(inputs_shared["latents"], device=pipe.device)
        image = pipe.vae_output_to_image(image)
        image.save(os.path.join(save_dir, f"{prefix}_{progress_id:03d}.png"))
        # Reload DIT for next step
        if progress_id < len(pipe.scheduler.timesteps) - 1:
            pipe.load_models_to_device(pipe.in_iteration_models)

    pipe.load_models_to_device([])
    return image


NICK_NAME = {
    "black-forest-labs/FLUX.1-dev": "flux.1-dev",
    "Wan-AI/Wan2.1-T2V-14B": "wan.1-t2v-14b",
    "Wan-AI/Wan2.2-T2V-A14B": "wan.2-t2v-a14b",
    "Qwen/Qwen-Image": "qwen-image-20b",
}


def get_prompts(n_prompts=10, seed=2025):
    response = requests.get(META_URL)
    data = response.json()
    prompts = []
    for img_url, value in data.items():
        prompts.append(value["prompt"])
    random.seed(seed)
    selected_prompts = random.sample(prompts, n_prompts)
    return selected_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.toml")
    parser.add_argument("--repo_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--num_frames", type=int, default=81)

    args = parser.parse_args()

    # Load configuration from TOML file
    config = toml.load(args.config)

    # Get model paths from configuration
    dfloat_dir = config["download"]["dfloat_dir"]
    dfloat_repo_id, dfloat_model_id = get_dfloat_model_name_or_path(
        args.repo_id, dfloat_dir
    )

    print(f"DFloat model path: {dfloat_repo_id}")
    print(f"Local model path: {args.repo_id}")

    # See if the row exists
    results_file = os.path.join(
        args.save_dir,
        f"{NICK_NAME[args.repo_id]}-seed{args.seed}-prompts{args.n_prompts}.csv",
    )
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            # Column names
            f.write(
                "no_compress,e2e_latency_mean,e2e_latency_std,step_latency_mean,step_latency_std,gpu_memory_allocated_mb,gpu_peak_memory_usage_mb,notes\n"
            )

    # Get no_compress_tag and notes
    no_compress_tag = "original" if args.no_compress else "compressed"
    if args.offload and no_compress_tag == "original":
        no_compress_tag = "original-offload"
    notes = "default"
    if args.repo_id in ["Wan-AI/Wan2.1-T2V-14B", "Wan-AI/Wan2.2-T2V-A14B"]:
        num_frames = args.num_frames
        if num_frames == 81:
            notes = "default"
            wan_height = 480
            wan_width = 832
        else:
            wan_height = 368
            wan_width = 640
            notes = f"{num_frames} frames with {wan_height}x{wan_width}"

    df = pd.read_csv(results_file)
    row = df[(df["no_compress"] == no_compress_tag) & (df["notes"] == notes)]
    if len(row) > 0:
        print(
            f"Results for {args.repo_id} with `no_compress` {no_compress_tag} and `notes` {notes} already exist in {results_file}"
        )
        print(row)
        exit()
    else:
        print(
            f"Getting new results for {args.repo_id} with `no_compress` {no_compress_tag} and `notes` {notes}"
        )

    if args.no_compress:
        if args.repo_id == "black-forest-labs/FLUX.1-dev":
            from diffsynth.pipelines.flux_image_new import (
                FluxImagePipeline,
                ModelConfig,
            )

            pipe = FluxImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(
                        model_id="black-forest-labs/FLUX.1-dev",
                        origin_file_pattern="flux1-dev.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="black-forest-labs/FLUX.1-dev",
                        origin_file_pattern="text_encoder/model.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="black-forest-labs/FLUX.1-dev",
                        origin_file_pattern="text_encoder_2/",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="black-forest-labs/FLUX.1-dev",
                        origin_file_pattern="ae.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                ],
            )
        elif args.repo_id == "Wan-AI/Wan2.1-T2V-14B":
            from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(
                        model_id="Wan-AI/Wan2.1-T2V-14B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Wan-AI/Wan2.1-T2V-14B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Wan-AI/Wan2.1-T2V-14B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                ],
            )
        elif args.repo_id == "Wan-AI/Wan2.2-T2V-A14B":
            from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(
                        model_id="Wan-AI/Wan2.2-T2V-A14B",
                        origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Wan-AI/Wan2.2-T2V-A14B",
                        origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Wan-AI/Wan2.2-T2V-A14B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Wan-AI/Wan2.2-T2V-A14B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                ],
            )
        elif args.repo_id == "Qwen/Qwen-Image":
            from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(
                        model_id=args.repo_id,
                        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id=args.repo_id,
                        origin_file_pattern="text_encoder/model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id=args.repo_id,
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                ],
                tokenizer_config=ModelConfig(
                    model_id=args.repo_id, origin_file_pattern="tokenizer/"
                ),
            )
        if args.offload:
            pipe.enable_vram_management(num_persistent_param_in_dit=0)
        else:
            pipe.enable_vram_management()
    else:
        pipe = DFloatDiffSynthModelFP8.from_pretrained(
            dfloat_repo_id, dfloat_model_id, args.repo_id
        )

    compress_or_not = "original" if args.no_compress else "compressed"
    save_dir = os.path.join(
        args.save_dir, NICK_NAME[args.repo_id], f"seed{args.seed}-{compress_or_not}"
    )
    os.makedirs(save_dir, exist_ok=True)

    prompts = get_prompts(n_prompts=args.n_prompts, seed=args.seed)
    with open(os.path.join(save_dir, f"prompts{args.n_prompts}.txt"), "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    latency_list = []
    with torch.no_grad():
        if args.repo_id == "black-forest-labs/FLUX.1-dev":
            num_inference_steps = 30

            # Warm-up
            image = pipe(
                prompt="An anime female character, character design, lofi style, soft colors, gentle natural linework, key art, emotion is happy. Hand drawn with an award-winning anime aesthetic. A well-defined nose. Holding a sign saying DFLOAT IS LOSSLESS",
                seed=args.seed,
                negative_prompt="Six fingers",
                num_inference_steps=num_inference_steps,
            )
            image.save(os.path.join(save_dir, "test.png"))
            torch.cuda.synchronize()

            for i, prompt in enumerate(prompts):
                start_time = time.time()
                intermediate_dir = os.path.join(save_dir, f"intermediates_{i}")
                image = generate_with_intermediate_images(
                    pipe,
                    prompt=prompt,
                    seed=args.seed,
                    num_inference_steps=num_inference_steps,
                    save_dir=intermediate_dir,
                    prefix=f"prompt{i}",
                )
                end_time = time.time()
                latency_list.append(end_time - start_time)
                image.save(os.path.join(save_dir, f"{i}.png"))
                torch.cuda.synchronize()

        elif args.repo_id == "Qwen/Qwen-Image":
            num_inference_steps = 40

            # Warm-up
            image = pipe(
                prompt="An anime female character, character design, lofi style, soft colors, gentle natural linework, key art, emotion is happy. Hand drawn with an award-winning anime aesthetic. A well-defined nose. Holding a sign saying DFLOAT IS LOSSLESS",
                seed=args.seed,
                num_inference_steps=num_inference_steps,
            )
            image.save(os.path.join(save_dir, "test.png"))
            torch.cuda.synchronize()

            for i, prompt in enumerate(prompts):
                start_time = time.time()
                image = pipe(
                    prompt=prompt,
                    seed=args.seed,
                    num_inference_steps=num_inference_steps,
                )
                end_time = time.time()
                latency_list.append(end_time - start_time)
                image.save(os.path.join(save_dir, f"{i}.png"))
                torch.cuda.synchronize()

        elif args.repo_id in ["Wan-AI/Wan2.1-T2V-14B", "Wan-AI/Wan2.2-T2V-A14B"]:
            num_inference_steps = 50

            # Warm-up
            negative_prompt = "Vivid colors, overexposure, static, unclear details, subtitles, style, artwork, painting, frame, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image, messy background, three legs, crowded background, walking backwards"
            video = pipe(
                prompt='An astronaut wearing a spacesuit rides a mechanical horse on the surface of Mars, facing the camera. The red desolate landscape extends into the distance, dotted with huge craters and strange rock formations. The mechanical horse"s gait is steady, kicking up faint dust, showcasing the perfect combination of future technology and primitive exploration. The astronaut holds a control device, with a determined gaze, as if pioneering humanity"s new frontier. The background features the deep universe and the blue Earth, creating a scene that is both sci-fi and full of hope, making one imagine future interstellar life.',
                negative_prompt=negative_prompt,
                seed=args.seed,
                tiled=True,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
            )
            save_video(
                video,
                os.path.join(save_dir, f"test-{num_frames}frames.mp4"),
                fps=15,
                quality=5,
            )
            torch.cuda.synchronize()

            for i, prompt in enumerate(prompts):
                start_time = time.time()
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=args.seed,
                    tiled=True,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    height=wan_height,  # 480
                    width=wan_width,  # 832
                )
                end_time = time.time()
                latency_list.append(end_time - start_time)
                save_video(
                    video,
                    os.path.join(save_dir, f"{i}-{num_frames}frames.mp4"),
                    fps=15,
                    quality=5,
                )
                torch.cuda.synchronize()

    # GPU memory tracking
    allocated = 0
    peak_allocated = 0
    for device_id in range(torch.cuda.device_count()):
        allocated += torch.cuda.memory_allocated(device_id)
        peak_allocated += torch.cuda.max_memory_allocated(device_id)

    allocated /= 1024**2  # Convert to MB
    peak_allocated /= 1024**2  # Convert to MB

    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Peak Memory Usage: {peak_allocated:.2f} MB")

    e2e_latency_list = np.array(latency_list)
    step_latency_list = e2e_latency_list / num_inference_steps
    e2e_latency_mean = np.mean(e2e_latency_list)
    e2e_latency_std = np.std(e2e_latency_list)
    step_latency_mean = np.mean(step_latency_list)
    step_latency_std = np.std(step_latency_list)

    # Save results
    with open(results_file, "a") as f:
        f.write(
            f"{no_compress_tag},{e2e_latency_mean:.4f},{e2e_latency_std:.4f},{step_latency_mean:.4f},{step_latency_std:.4f},{allocated:.2f},{peak_allocated:.2f},{notes}\n"
        )
