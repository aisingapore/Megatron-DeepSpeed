import argparse
import json
import os
import os.path
import random
import re

import numpy as np
import torch
from transformers import LlamaConfig

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

TRANSFORMERS_TO_MEGATRON = {
    "self_attn.dense": "self_attention.dense",
    "mlp.dense_h_to_4h_1": "mlp.dense_h_to_4h_1",
    "mlp.dense_h_to_4h_2": "mlp.dense_h_to_4h_2",
    "mlp.dense_4h_to_h": "mlp.dense_4h_to_h",
}

TENSOR_PARALLEL_PARAMS = [
    # megatron-lm layers to merge across tp ranks
    "self_attn.query.weight",
    "self_attn.key_value.weight",
    "self_attn.dense.weight",
    "mlp.dense_h_to_4h_1.weight",
    "mlp.dense_h_to_4h_2.weight",
    "mlp.dense_4h_to_h.weight"
]


def add_checkpointing_args(parser):
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument("--load_path", type=str, required=True, help="Path to the checkpoint to convert.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the converted checkpoint.")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_num_key_value_heads",
        type=int,
        default=4,
        help=(
            "Number of key value heads for GQA. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_checkpoint_version",
        type=int,
        default=3,
        help=(
            "Megatron checkpoint version. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_vocab_size",
        type=int,
        default=128000,
        help=(
            "Megatron checkpoint vocab size. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument('--extra_num_vocabs', type=int, default=0)

    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def load_transformers_sharded_states(path):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.
    Only merges pytorch_model.bin.
    Args:
        path (str): the path to the sharded checkpoints
    """
    index_path = os.path.join(path, "pytorch_model.bin.index.json")
    with open(index_path, "r") as infile:
        data = json.load(infile)

    state_dict = {}
    for shard_path in set(data["weight_map"].values()):
        chunk = torch.load(os.path.join(path, shard_path), map_location="cpu")
        state_dict.update(chunk)

    return state_dict


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def megatron_to_transformers_fix_query_key_value_output_ordering(
    output, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Rearrange and reshape Megatron output (seq_len, batch, num_split * num_head * hidden_size) to transformers output
    (batch, seq_len, num_split * num_head * hidden_size).

    This function is used to verify that spliting the key_value weights is
    done correctly during development. Not used in the actual conversion.
    
    Args:
        output (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # output is (seq_len, batch, num_split * num_head * hidden_size)
    input_shape = output.size()
    if checkpoint_version == 1.0:
        raise NotImplementedError
    elif checkpoint_version >= 2.0:
        saved_shape = input_shape[:2] + (num_heads, num_splits, hidden_size)
        output = output.view(*saved_shape)
        output = output.transpose(2, 3).contiguous()
    output = output.view(*input_shape).transpose(0, 1)
    return output


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def meanpool_mha_to_gqa(param, checkpoint_version, num_attention_heads, num_key_value_heads, hidden_size):
    """
    Mean pools MHA weights to GQA weights. Expects input shape to be
    (num_heads * num_heads * num
    Reference: https://github.com/NVIDIA/Megatron-LM/issues/480
    Args:
        param (torch.Tensor): the tensor to meanpool
        checkpoint_version (int): the version of the checkpoint
        num_attention_heads (int): the number of attention heads for MHA
        num_key_value_heads (int): the number of attention heads for GQA
        hidden_size (int): the hidden size per head
    """
    input_shape = param.size()
    if checkpoint_version == 1.0:
        raise NotImplementedError
    elif checkpoint_version >= 2.0:
        num_key_value_groups = num_attention_heads // num_key_value_heads
        mha_shape = (num_key_value_heads, num_key_value_groups, hidden_size) + input_shape[1:]
        gqa_shape = (num_key_value_heads * hidden_size,) + input_shape[1:]
        param = torch.mean(param.view(*mha_shape), dim=1)
    param = param.view(*gqa_shape)
    return param


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.
    Args:
        args (argparse.Namespace): the arguments to the script
    """
    os.makedirs(args.save_path, exist_ok=True)

    state_dict = load_transformers_sharded_states(args.load_path)
    config = LlamaConfig.from_pretrained(args.load_path)
    # MHA to GQA related configs
    head_dim = config.hidden_size // config.num_attention_heads
    
    internal_state_dict = {}
    
    for layer_id in range(config.num_hidden_layers):
        # Attention
        q_weight = state_dict[f"model.layers.{layer_id}.self_attn.q_proj.weight"]
        k_weight = meanpool_mha_to_gqa(
            state_dict[f"model.layers.{layer_id}.self_attn.k_proj.weight"],
            args.target_checkpoint_version,
            config.num_attention_heads,
            args.target_num_key_value_heads,
            head_dim,
        )
        v_weight = meanpool_mha_to_gqa(
            state_dict[f"model.layers.{layer_id}.self_attn.v_proj.weight"],
            args.target_checkpoint_version,
            config.num_attention_heads,
            args.target_num_key_value_heads,
            head_dim,
        )
    
        internal_state_dict[f"transformer.layers.{layer_id}.self_attn.query.weight"] = q_weight
        internal_state_dict[f"transformer.layers.{layer_id}.self_attn.key_value.weight"] = torch.cat((k_weight, v_weight))
        internal_state_dict[f"transformer.layers.{layer_id}.self_attn.dense.weight"] =\
            state_dict[f"model.layers.{layer_id}.self_attn.o_proj.weight"]
    
        # FFN
        dense_h_to_4h_1_weight = state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"]
        dense_h_to_4h_2_weight = state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"]
    
        internal_state_dict[f"transformer.layers.{layer_id}.mlp.dense_h_to_4h_1.weight"] =\
            dense_h_to_4h_1_weight
        internal_state_dict[f"transformer.layers.{layer_id}.mlp.dense_h_to_4h_2.weight"] =\
            dense_h_to_4h_2_weight
        internal_state_dict[f"transformer.layers.{layer_id}.mlp.dense_4h_to_h.weight"] =\
            state_dict[f"model.layers.{layer_id}.mlp.down_proj.weight"]
    
        # Layernorm
        internal_state_dict[f"transformer.layers.{layer_id}.input_layernorm.weight"] =\
            state_dict[f"model.layers.{layer_id}.input_layernorm.weight"]
        internal_state_dict[f"transformer.layers.{layer_id}.post_attention_layernorm.weight"] =\
            state_dict[f"model.layers.{layer_id}.post_attention_layernorm.weight"]
    
    internal_state_dict["transformer.embedding.weight"] = state_dict["model.embed_tokens.weight"]
    internal_state_dict["transformer.embedding.word_embeddings.weight"] = state_dict["model.embed_tokens.weight"]
    internal_state_dict["transformer.final_layernorm.weight"] = state_dict["model.norm.weight"]
    internal_state_dict["transformer.lm_head.weight"] = state_dict["lm_head.weight"]
    state_dict = internal_state_dict

    # Save the tracker file
    tracker_path = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_path, "w") as outfile:
        outfile.write("release")
    
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = argparse.Namespace(
        orig_vocab_size=config.vocab_size,
        vocab_size=args.target_vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=args.target_num_key_value_heads,
        tensor_model_parallel_size=args.target_tensor_model_parallel_size,
        pipeline_model_parallel_size=args.target_pipeline_model_parallel_size,
        data_parallel_size=args.target_data_parallel_size,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        rank=0,
        tokenizer_type="GPT2BPETokenizer",
    )
    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(megatron_args, "params_dtype", dtype)

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # Embedding layer
    print("Converting embedding layer")
    word_embedding = state_dict["transformer.embedding.word_embeddings.weight"].to(dtype)
    lm_head = state_dict["transformer.lm_head.weight"].to(dtype)
    
    # Discard pretrained embeddings, initialize with new vocab size
    full_word_embed = torch.nn.Embedding(megatron_args.vocab_size, word_embedding.shape[1]).weight
    full_lm_head = torch.nn.Linear(lm_head.shape[1], megatron_args.vocab_size).weight
    
    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i]
    
    out_lm_head = torch.chunk(full_lm_head, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        lm_head_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.lm_head"
        )
        lm_head_dict["weight"] = out_lm_head[i]

    # Transformer layers
    print("Converting transformer layers")
    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism "
            f"({args.target_pipeline_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size
    
    layer_re = re.compile("transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    num_attention_heads = megatron_args.num_attention_heads
    num_key_value_heads = megatron_args.num_key_value_heads
    
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})
    
        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"transformer.layers.{pp_layer_id}.")
            ]
    
            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break
    
                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight = m.group(3)
    
                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.startswith("input_layernorm") or op_name.startswith("post_attention_layernorm"):
                    out_name = "input_layernorm" if op_name.endswith("input_layernorm") else "post_attention_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.{weight}"
    
                # TODO: Original implementation copied rotary_emb to each attention layer. Check that we copy it to args.
    
                # handle attention K, V, Q weights
                elif op_name.startswith("self_attn.query") and weight == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        args.target_checkpoint_version,
                        1,
                        num_attention_heads,
                        head_dim,
                    )
                    layer_name = f"layers.{layer}.self_attention.query.{weight}"
                elif op_name.startswith("self_attn.key_value") and weight == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        args.target_checkpoint_version,
                        2,
                        num_key_value_heads,
                        head_dim,
                    )
                    layer_name = f"layers.{layer}.self_attention.key_value.{weight}"
    
                # handle attention and mlp weights
                elif weight == "weight":
                    out_name = TRANSFORMERS_TO_MEGATRON.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f"layers.{layer}.{out_name}.{weight}"
    
                # skip
                else:
                    continue
    
                # Split tensor model parallel parameters
                if f"{op_name}.{weight}" in TENSOR_PARALLEL_PARAMS:
                    # self_attn.dense and mlp.dense_4h_to_h are RowParallelLinear
                    # the rest are ColumnParallelLinear
                    dim = 1 if op_name in ["self_attn.dense", "mlp.dense_4h_to_h"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)
    
                for i in range(args.target_tensor_model_parallel_size):
                    # get_element_from_dict_by_path will recursively create empty dictionaries if the specified path does not exist.
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if f"{op_name}.{weight}" in TENSOR_PARALLEL_PARAMS else params.clone()
                    )
    
            # Process layers which requires additional processing
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
    
                # FFN
                dense_h_to_4h_1_name = "mlp.dense_h_to_4h_1.weight"
                dense_h_to_4h_1_layer_name = f"layers.{layer}.{dense_h_to_4h_1_name}"
                dense_h_to_4h_1_weight = params_dict[dense_h_to_4h_1_layer_name]
    
                dense_h_to_4h_2_name = "mlp.dense_h_to_4h_2.weight"
                dense_h_to_4h_2_layer_name = f"layers.{layer}.{dense_h_to_4h_2_name}"
                dense_h_to_4h_2_weight = params_dict[dense_h_to_4h_2_layer_name]
    
                dense_h_to_4h_name = "mlp.dense_h_to_4h.weight"
                dense_h_to_4h_layer_name = f"layers.{layer}.{dense_h_to_4h_name}"
    
                # Concat gate_proj and up_proj into a single tensor
                params_dict[dense_h_to_4h_layer_name] = torch.cat([dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)
                # Clean up
                del params_dict[dense_h_to_4h_1_layer_name]
                del params_dict[dense_h_to_4h_2_layer_name]
    
                # TODO: Original implementation merged q and kv into a single qkv. Check if we get the same output

    if pp_rank == args.target_pipeline_model_parallel_size - 1:
        # handle final layernorm
        for weight_or_bias in ["weight"]:
            params = state_dict[f"transformer.final_layernorm.{weight_or_bias}"].to(dtype)
            layer_name = f"final_layernorm.{weight_or_bias}"
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                params_dict[layer_name] = params.clone()

        # add embedding
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.embedding")
            params_dict["weight"] = out_word_embed[i].clone()
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.embedding.word_embeddings")
            params_dict["weight"] = out_word_embed[i].clone()

        # add the LM head
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.output_layer")
            params_dict["weight"] = out_lm_head[i].clone()

    # saving the state dict as per the tp_rank and pp_rank
    for tp_rank in range(args.target_tensor_model_parallel_size):
        output_state_dict[tp_rank]["checkpoint_version"] = float(args.target_checkpoint_version)
        output_state_dict[tp_rank]["args"] = megatron_args
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
            if args.target_pipeline_model_parallel_size == 1
            else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )

        checkpoint_name = "model_optim_rng.pt"
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if args.print_checkpoint_structure:
            print(
                f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} "
                f"and PP rank {pp_rank}:"
            )
            recursive_print(None, output_state_dict[tp_rank])
        torch.save(output_state_dict[tp_rank], checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_checkpointing_args(parser)
    add_megatron_checkpoint_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_transformers_to_megatron(args)

