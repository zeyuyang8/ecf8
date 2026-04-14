import json
import math
import os
import re
from sys import stderr
from typing import Dict, Optional, Union

import cupy as cp
import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from diffsynth.core.vram.layers import AutoWrappedLinear
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from ..const import get_default_no_split_pattern
from ..utils import report_model_size

# # Load CUDA kernel for custom DFloat decoding
# ptx_path = os.path.join(os.path.dirname(__file__), 'decode.ptx')
# _decode = cp.RawModule(path=ptx_path).get_function('decode')

# Load CUDA kernel for custom DFloat decoding
kernel_path = os.path.join(os.path.dirname(__file__), "decode.cu")
with open(kernel_path, "r") as f:
    _decode_kernel = f.read()
    options = ("--generate-line-info",)
    _decode = cp.RawKernel(
        _decode_kernel,
        "decode",
        options=options,
    )


class TensorManager:
    # Static class variable to store tensors for each device
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def get_tensor(device, n_elements, dtype=torch.float8_e4m3fn):
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)

        # Check if we already have a tensor for this device
        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]

            # If existing tensor is large enough, return a slice of it
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]

            # Otherwise, delete the existing tensor to free up memory
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()  # Ensure memory is freed

        # Allocate a new tensor
        new_tensor = torch.empty(n_elements, dtype=dtype, device=device)
        print(f"Allocated {n_elements} {dtype} on device {device}", file=stderr)

        # Store the tensor
        TensorManager._tensors[device] = new_tensor

        return new_tensor

    @staticmethod
    def clear_device(device=None):
        if device is None:
            # Clear all devices
            TensorManager._tensors.clear()
        else:
            # Convert device to torch.device if it's a string
            if isinstance(device, str):
                device = torch.device(device)

            # Remove specific device
            if device in TensorManager._tensors:
                del TensorManager._tensors[device]

        torch.cuda.empty_cache()  # Ensure memory is freed


def get_hook(threads_per_block, bytes_per_thread):
    threads_per_block = (threads_per_block,)

    def decode_hook(module, _):
        # Get dimensions for tensor reconstruction
        n_elements = module.packed_other_4bits.numel() * 2
        n_bytes = module.encoded.numel()
        n_luts = module.luts.shape[0]

        # Get output tensor for reconstructed weights
        device = module.encoded.device
        reconstructed = TensorManager.get_tensor(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (
            int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))),
        )

        # Launch CUDA kernel to decode the compressed weights
        with cp.cuda.Device(device.index):
            _decode(
                grid=blocks_per_grid,
                block=threads_per_block,
                shared_mem=module.shared_mem_size,
                args=[
                    module.luts.data_ptr(),
                    module.encoded.data_ptr(),
                    module.packed_other_4bits.data_ptr(),
                    module.output_positions.data_ptr(),
                    module.gaps.data_ptr(),
                    reconstructed.data_ptr(),
                    n_luts,
                    n_bytes,
                    n_elements,
                ],
            )

        # Inject reconstructed weights into the appropriate module
        if isinstance(module, nn.Linear):
            module.weight = reconstructed.view(module.out_features, module.in_features)
        elif isinstance(module, nn.Embedding):
            module.weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        else:
            # Handle special case where weights need to be split across multiple submodules
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                if isinstance(sub_module, AutoWrappedLinear) or isinstance(
                    sub_module, nn.Linear
                ):
                    try:
                        sub_module.weight = weight.view(
                            sub_module.out_features, sub_module.in_features
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"sub_module: {sub_module}")
                        print(f"weight: {weight}")
                        print(f"weight.shape: {weight.shape}")
                        print(f"module.split_positions: {module.split_positions}")
                        print(f"reconstructed.shape: {reconstructed.shape}")
                        print(f"module {module}")
                        raise e
                elif isinstance(sub_module, nn.Embedding):
                    sub_module.weight = weight.view(
                        sub_module.num_embeddings, sub_module.embedding_dim
                    )
                else:
                    raise ValueError(f"Unsupported module type: {type(sub_module)}")

    return decode_hook


def load_and_replace_tensors(model, directory_path, dfloat_config):
    threads_per_block = dfloat_config["threads_per_block"]
    bytes_per_thread = dfloat_config["bytes_per_thread"]
    pattern_dict = dfloat_config["pattern_dict"]["fp8"]
    # print(pattern_dict, file=stderr)

    # Get all .safetensors files in the directory
    safetensors_files = [
        f for f in os.listdir(directory_path) if f.endswith(".safetensors")
    ]
    for file_name in tqdm(safetensors_files, desc="Loading DFloat safetensors"):
        file_path = os.path.join(directory_path, file_name)

        # Load the tensors from the file
        loaded_tensors = load_file(file_path)

        # Iterate over each tensor in the file
        for tensor_name, tensor_value in loaded_tensors.items():
            # Check if this tensor exists in the model's state dict
            if tensor_name in model.state_dict():
                # Get the parameter or buffer
                if tensor_name in dict(model.named_parameters()):
                    # It's a parameter, we can set it directly
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        print(
                            f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}",
                            file=stderr,
                        )
                else:
                    # It's a buffer, we can also set it directly
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        print(
                            f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}",
                            file=stderr,
                        )
            else:
                # Split the tensor name to get module path
                parts = tensor_name.split(".")
                module = model

                # Navigate to the correct module
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        print(f"Cannot find module path for {tensor_name}", file=stderr)
                        break
                else:
                    if parts[-1] == "split_positions":
                        setattr(module, "split_positions", tensor_value.tolist())
                    else:
                        # Register the buffer to the found module
                        module.register_buffer(parts[-1], tensor_value)

                    # Set up decompression for encoded weights
                    if parts[-1] == "encoded":
                        # Register the decode hook to decompress weights during forward pass
                        module.register_forward_pre_hook(
                            get_hook(threads_per_block, bytes_per_thread)
                        )

                        # Configure weight injection based on module type
                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, ".".join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    # Remove weight attribute from embedding layer
                                    tmp = module.weight
                                    delattr(module, "weight")
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    # Remove weight attribute from linear layer
                                    tmp = module.weight
                                    delattr(module, "weight")
                                    del tmp
                                else:
                                    # Handle special case for multi-module weight injection
                                    setattr(module, "weight_injection_modules", [])
                                    for attr_path in attr_names:
                                        parts = attr_path.split(".")
                                        target = module
                                        for p in parts:
                                            target = getattr(target, p)

                                        if hasattr(target, "weight"):
                                            tmp = target.weight
                                            delattr(target, "weight")
                                            del tmp
                                        else:
                                            raise ValueError(
                                                f"Weight not found for {attr_path}"
                                            )
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == "output_positions":
                        output_positions_np = module.output_positions.view(
                            torch.uint64
                        ).numpy()

                        # Calculate how many elements each thread block is responsible for
                        elements_per_block = (
                            output_positions_np[1:] - output_positions_np[:-1]
                        )

                        # Find the maximum number of elements any block will decode
                        max_elements_per_block = elements_per_block.max().item()
                        shared_mem_size = (
                            threads_per_block * 4 + 4 + max_elements_per_block * 1
                        )

                        # Calculate required shared memory size for CUDA kernel
                        setattr(module, "shared_mem_size", shared_mem_size)

    return model


def load_and_replace_tensors_parallel(model, directory_path, dfloat_config):
    import concurrent.futures
    import gc
    from concurrent.futures import ThreadPoolExecutor

    threads_per_block = dfloat_config["threads_per_block"]
    bytes_per_thread = dfloat_config["bytes_per_thread"]
    pattern_dict = dfloat_config["pattern_dict"]["fp8"]

    # PRE-COMPUTE these expensive lookups once (MAJOR speedup)
    print("Pre-computing model structure...", file=stderr)
    model_state_dict = model.state_dict()
    model_parameters = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    module_cache = {}  # Cache for module lookups

    def get_module_fast(tensor_name):
        parts = tensor_name.split(".")
        module_path = ".".join(parts[:-1])

        if module_path in module_cache:
            return module_cache[module_path], parts[-1]

        module = model
        for part in parts[:-1]:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                print(f"Cannot find module path for {tensor_name}", file=stderr)
                return None, None

        module_cache[module_path] = module
        return module, parts[-1]

    def process_tensor_immediate(tensor_name, tensor_value):
        if tensor_name in model_state_dict:
            if tensor_name in model_parameters:
                param = model_parameters[tensor_name]
                if param.shape == tensor_value.shape:
                    param.data = tensor_value
                else:
                    print(
                        f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}",
                        file=stderr,
                    )
            else:
                buffer = model_buffers[tensor_name]
                if buffer.shape == tensor_value.shape:
                    buffer.data = tensor_value
                else:
                    print(
                        f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}",
                        file=stderr,
                    )
        else:
            # Process compressed tensors
            module, attr_name = get_module_fast(tensor_name)
            if module is not None:
                _process_compressed_tensor(
                    module,
                    attr_name,
                    tensor_value,
                    tensor_name,
                    pattern_dict,
                    threads_per_block,
                    bytes_per_thread,
                )

    def load_and_process_file(file_name):
        file_path = os.path.join(directory_path, file_name)
        try:
            loaded_tensors = load_file(file_path)
            processed_count = 0

            # Process each tensor immediately
            for tensor_name, tensor_value in loaded_tensors.items():
                process_tensor_immediate(tensor_name, tensor_value)
                processed_count += 1

                # Periodic garbage collection for very large models
                if processed_count % 100 == 0:
                    gc.collect()

            return len(loaded_tensors)
        except Exception as e:
            print(f"Error loading {file_name}: {e}", file=stderr)
            return 0

    # Get all .safetensors files
    safetensors_files = [
        f for f in os.listdir(directory_path) if f.endswith(".safetensors")
    ]

    # For very large models, use fewer workers to control memory usage
    total_files = len(safetensors_files)
    if total_files > 20:  # Likely a very large model
        max_workers = min(4, os.cpu_count() or 2)  # Conservative for memory
        print(
            f"Large model detected ({total_files} files), using {max_workers} workers",
            file=stderr,
        )
    else:
        max_workers = min(8, total_files, os.cpu_count() or 4)

    # Process files with controlled parallelism
    total_tensors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_and_process_file, f): f for f in safetensors_files
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(safetensors_files),
            desc="Loading DFloat safetensors",
        ):
            tensor_count = future.result()
            total_tensors += tensor_count

            # Force garbage collection after each file for large models
            if total_files > 50:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"Processed {total_tensors} tensors from {total_files} files", file=stderr)
    return model


def _process_compressed_tensor(
    module,
    attr_name,
    tensor_value,
    tensor_name,
    pattern_dict,
    threads_per_block,
    bytes_per_thread,
):
    if attr_name == "split_positions":
        setattr(module, "split_positions", tensor_value.tolist())
    else:
        # Register the buffer to the found module
        module.register_buffer(attr_name, tensor_value)

    # Set up decompression for encoded weights
    if attr_name == "encoded":
        # Register the decode hook to decompress weights during forward pass
        module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))

        # Configure weight injection based on module type
        module_path = ".".join(tensor_name.split(".")[:-1])
        for pattern, attr_names in pattern_dict.items():
            if re.fullmatch(pattern, module_path):
                if isinstance(module, nn.Embedding):
                    tmp = module.weight
                    delattr(module, "weight")
                    del tmp
                elif isinstance(module, nn.Linear):
                    tmp = module.weight
                    delattr(module, "weight")
                    del tmp
                else:
                    # Handle special case for multi-module weight injection
                    setattr(module, "weight_injection_modules", [])
                    for attr_path in attr_names:
                        parts = attr_path.split(".")
                        target = module
                        for p in parts:
                            target = getattr(target, p)

                        tmp = target.weight
                        delattr(target, "weight")
                        del tmp
                        module.weight_injection_modules.append(target)
                break

    elif attr_name == "output_positions":
        output_positions_np = module.output_positions.view(torch.uint64).numpy()
        elements_per_block = output_positions_np[1:] - output_positions_np[:-1]
        max_elements_per_block = elements_per_block.max().item()
        shared_mem_size = threads_per_block * 4 + 4 + max_elements_per_block * 1
        setattr(module, "shared_mem_size", shared_mem_size)


def get_no_split_classes(model, patterns):
    no_split_classes = []
    for pattern in patterns:
        for full_name, sub_module in model.named_modules():
            if re.fullmatch(pattern, full_name):
                class_name = sub_module.__class__.__name__
                if class_name not in no_split_classes:
                    no_split_classes.append(class_name)

    return no_split_classes


def get_dfloat_model_name_or_path(repo_id: str, dfloat_dir: str = "../dfloat-models"):
    split_repo_id = repo_id.split("/")
    split_repo_id[0] = "DFloat11"
    dfloat_model_id = "/".join(split_repo_id) + "-DF6.5"
    dfloat_repo_id = os.path.join(
        dfloat_dir, "models--" + "--".join(split_repo_id) + "-DF6.5"
    )
    return dfloat_repo_id, dfloat_model_id


class DFloatModel:
    @classmethod
    def from_pretrained(
        cls,
        dfloat_repo_id: str,
        dfloat_model_id: str,
        model_name_or_path: str,
        device: Optional[str] = None,
        device_map: str = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        if not os.path.exists(dfloat_repo_id):
            snapshot_download(
                repo_id=dfloat_model_id,
                local_dir=dfloat_repo_id,
            )

        # Load model with proper layer types (e.g. FP8Linear for FP8 models)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            attn_implementation=kwargs.get("attn_implementation"),
            low_cpu_mem_usage=True,
        )
        model.eval()
        print(f"Model device: {model.device}")

        # Verify model has DFloat configuration
        config = AutoConfig.from_pretrained(dfloat_repo_id)
        assert hasattr(config, "dfloat_config")
        dfloat_config = config.dfloat_config
        # Read JSON file
        pattern_dict_path = os.path.join(dfloat_repo_id, "pattern_dict.json")
        assert os.path.exists(pattern_dict_path)
        with open(pattern_dict_path, "r") as f:
            pattern_dict = json.load(f)
            dfloat_config["pattern_dict"] = pattern_dict

        # bf16_patterns = list(pattern_dict['bf16'].keys())
        fp8_patterns = pattern_dict["fp8"]
        include_gb, exclude_gb, model_gb = report_model_size(
            model, include_pattern=fp8_patterns
        )
        with open(os.path.join(dfloat_repo_id, "model_size.csv"), "w") as f:
            f.write("fp8_blocks,rest,total\n")
            f.write(f"{include_gb},{exclude_gb},{model_gb}\n")

        # Load compressed weights and configure decompression
        load_and_replace_tensors_parallel(model, dfloat_repo_id, dfloat_config)
        _, _, model_gb = report_model_size(model, include_pattern=None)
        include_gb = model_gb - exclude_gb
        with open(os.path.join(dfloat_repo_id, "model_size.csv"), "a") as f:
            f.write(f"{include_gb},{exclude_gb},{model_gb}\n")

        # Move model to specified device or distribute across multiple devices
        if device:
            model = model.to(device)
        else:
            assert device_map == "auto", (
                'device_map should be "auto" if no specific device is provided.'
            )
            # Identify modules that must remain on same device for decompression
            model_class = model.__class__.__name__
            patterns = get_default_no_split_pattern(model_class)
            no_split_classes = get_no_split_classes(model, patterns)
            max_memory = get_balanced_memory(
                model, max_memory=max_memory, no_split_module_classes=no_split_classes
            )
            device_map = infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=no_split_classes
            )
            model = dispatch_model(model, device_map)

        # Warn if model is not fully on GPU
        if any(param.device.type == "cpu" for param in model.parameters()):
            print(
                "Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.",
                file=stderr,
            )

        return model


class DFloatDiffSynthModelFP8:
    @classmethod
    def from_pretrained(
        cls,
        dfloat_repo_id: str,
        dfloat_model_id: str,
        model_name_or_path: str,
        device: Optional[str] = "cuda",
    ):
        if model_name_or_path == "black-forest-labs/FLUX.1-dev":
            from diffsynth.pipelines.flux_image import (
                FluxImagePipeline,
                ModelConfig,
            )

            pipe = FluxImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
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
        elif model_name_or_path == "Wan-AI/Wan2.1-T2V-14B":
            from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
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
        elif model_name_or_path == "Wan-AI/Wan2.2-T2V-A14B":
            from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
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
        elif model_name_or_path == "Qwen/Qwen-Image":
            from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=[
                    ModelConfig(
                        model_id="Qwen/Qwen-Image",
                        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Qwen/Qwen-Image",
                        origin_file_pattern="text_encoder/model*.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                    ModelConfig(
                        model_id="Qwen/Qwen-Image",
                        origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                        offload_dtype=torch.float8_e4m3fn,
                    ),
                ],
                tokenizer_config=ModelConfig(
                    model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"
                ),
            )

        pipe.device = device
        pipe.enable_vram_management(num_persistent_param_in_dit=0)

        pattern_dict_path = os.path.join(dfloat_repo_id, "pattern_dict.json")
        dfloat_config_path = os.path.join(dfloat_repo_id, "dfloat_config.json")

        with open(dfloat_config_path, "r") as f:
            dfloat_config = json.load(f)

        with open(pattern_dict_path, "r") as f:
            pattern_dict = json.load(f)

        dfloat_config["pattern_dict"] = pattern_dict

        fp8_patterns = pattern_dict["fp8"]
        include_gb, exclude_gb, model_gb = report_model_size(
            pipe, include_pattern=fp8_patterns
        )
        with open(os.path.join(dfloat_repo_id, "model_size.csv"), "w") as f:
            f.write("fp8_blocks,rest,total\n")
            f.write(f"{include_gb},{exclude_gb},{model_gb}\n")

        # Load DFloat weights AFTER enabling VRAM management
        load_and_replace_tensors_parallel(pipe, dfloat_repo_id, dfloat_config)
        _, _, model_gb = report_model_size(pipe, include_pattern=None)
        include_gb = model_gb - exclude_gb
        with open(os.path.join(dfloat_repo_id, "model_size.csv"), "a") as f:
            f.write(f"{include_gb},{exclude_gb},{model_gb}\n")

        pipe.to(device)
        pipe.eval()

        if any(param.device.type == "cpu" for param in pipe.parameters()):
            print(
                "Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.",
                file=stderr,
            )

        return pipe
