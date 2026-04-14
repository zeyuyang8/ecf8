FP8_PATTERN_DICT = {
    "llm": {
        "bf16": [r"model\.embed_tokens", r"lm_head"],
        "fp8": [r"model\.layers\.\d+"],
    },
    "deepseekv3": {
        "bf16": [r"model\.embed_tokens", r"lm_head"],
        "fp8": [
            r"model\.layers\.[0-2]",
            r"model\.layers\.(?![0-2](?:[^0-9]|$))\d+\.self_attn",
            r"model\.layers\.(?![0-2](?:[^0-9]|$))\d+\.mlp\.shared_experts",
            r"model\.layers\.(?![0-2](?:[^0-9]|$))\d+\.mlp\.experts\.\d+",
        ],
    },
    "qwen3moe": {
        "bf16": [r"model\.embed_tokens", r"lm_head"],
        "fp8": [
            r"model\.layers\.\d+\.self_attn",
            r"model\.layers\.\d+\.mlp\.experts\.\d+",
        ],
    },
    "diffsynth-flux-fp8": {
        "fp8": {
            r"text_encoder_2\.encoder\.block\.\d+": [
                "layer.0.SelfAttention.k",
                "layer.0.SelfAttention.o",
                "layer.0.SelfAttention.q",
                "layer.0.SelfAttention.v",
            ],
            r"dit\.blocks\.\d+": [
                "attn.a_to_out",
                "attn.a_to_qkv",
                "attn.b_to_out",
                "attn.b_to_qkv",
                "ff_a.0",
                "ff_a.2",
                "ff_b.0",
                "ff_b.2",
            ],
            r"dit\.single_blocks\.\d+": [
                "norm.linear",
                "proj_out",
                "to_qkv_mlp",
            ],
        },
        "bf16": [],
    },
    "diffsynth-wanvideo-fp8": {
        "fp8": {
            r"text_encoder\.blocks\.\d+": {
                "attn.k",
                "attn.o",
                "attn.q",
                "attn.v",
                "ffn.fc1",
                "ffn.fc2",
                "ffn.gate.0",
            },
            r"dit\.blocks\.\d+": {
                "cross_attn.k",
                "cross_attn.o",
                "cross_attn.q",
                "cross_attn.v",
                "ffn.0",
                "ffn.2",
                "self_attn.k",
                "self_attn.o",
                "self_attn.q",
                "self_attn.v",
            },
            r"dit2\.blocks\.\d+": {
                "cross_attn.k",
                "cross_attn.o",
                "cross_attn.q",
                "cross_attn.v",
                "ffn.0",
                "ffn.2",
                "self_attn.k",
                "self_attn.o",
                "self_attn.q",
                "self_attn.v",
            },
        },
        "bf16": [],
    },
    "diffsynth-qwenimage-fp8": {
        "fp8": [
            r"text_encoder\.lm_head",
            r"text_encoder\.model\.visual",
            r"text_encoder\.model\.language_model\.layers\.\d+",
            r"dit\.transformer_blocks\.\d+",
        ],
        "bf16": [],
    },
}

MODEL_CLASS_MAP = {
    "llm": set(
        [
            "Qwen3ForCausalLM",
            "LlamaForCausalLM",
        ]
    ),
    "deepseekv3": set(
        [
            "DeepseekV3ForCausalLM",
        ]
    ),
    "qwen3moe": set(
        [
            "Qwen3MoeForCausalLM",
        ]
    ),
    "diffsynth-flux-fp8": set(
        [
            "FluxImagePipeline",
        ]
    ),
    "diffsynth-wanvideo-fp8": set(
        [
            "WanVideoPipeline",
        ]
    ),
    "diffsynth-qwenimage-fp8": set(
        [
            "QwenImagePipeline",
        ]
    ),
}

DEFAULT_NO_SPLIT_PATTERN = {
    "llm": [r"model\.embed_tokens", r"model\.layers\.\d+", r"lm_head"],
    "deepseekv3": [r"model\.embed_tokens", r"model\.layers\.\d+", r"lm_head"],
    "qwen3moe": [r"model\.embed_tokens", r"model\.layers\.\d+", r"lm_head"],
    "diffsynth-flux-fp8": [
        r"text_encoder_2\.encoder\.block\.\d+",
        r"dit\.blocks\.\d+",
        r"dit\.single_blocks\.\d+",
    ],
    "diffsynth-wanvideo-fp8": [
        r"text_encoder\.blocks\.\d+",
        r"dit\.blocks\.\d+",
        r"dit2\.blocks\.\d+",
    ],
    "diffsynth-qwenimage-fp8": [
        r"text_encoder\.lm_head",
        r"text_encoder\.model\.visual",
        r"text_encoder\.model\.language_model\.layers\.\d+",
        r"dit\.transformer_blocks\.\d+",
    ],
}


def get_fp8_pattern_dict(model_class):
    for key, value in FP8_PATTERN_DICT.items():
        if model_class in MODEL_CLASS_MAP[key]:
            return value

    raise ValueError(f"Model class {model_class} not supported")


def get_default_no_split_pattern(model_class):
    for key, value in DEFAULT_NO_SPLIT_PATTERN.items():
        if model_class in MODEL_CLASS_MAP[key]:
            return value

    raise ValueError(f"Model class {model_class} not supported")
