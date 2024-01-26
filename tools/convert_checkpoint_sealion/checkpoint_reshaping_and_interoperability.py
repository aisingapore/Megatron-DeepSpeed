# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import json
import os
import re
import sys
import types
import numpy as np
import torch
import shutil

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from transformers import AutoTokenizer, GPT2Config, LlamaConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint

from megatron.tokenizer.sealion_tokenization import SEALIONTokenizer

def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )
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
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        '--extra_num_vocabs',
        type=int,
        default=0,
    )

    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_h_to_4h": [".mlp.gate_proj.",".mlp.up_proj."],
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
    "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq"
}

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query.weight",
    "self_attention.key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight"
]

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


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


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

def _init_embedding_weights(module):
    std = 0.02
    module.weight.data.normal_(mean=0.0, std=std)


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # # # Saving config and tokenzier files
    # # config_path = '/'.join(args.load_path.split('/')[:-1])
    # # os.system("cp -rf "+config_path+"/*.json " + args.save_path)
    # # os.system("cp -rf " + config_path + "/tokenizer.model " + args.save_path)

    # vocab_size = (
    #     megatron_args.padded_vocab_size
    #     if getattr(megatron_args, "orig_vocab_size", None) is None
    #     else megatron_args.orig_vocab_size
    # )

    # # params dtype
    # if args.target_params_dtype == "fp16":
    #     dtype = torch.float16
    # elif args.target_params_dtype == "bf16":
    #     dtype = torch.bfloat16
    # else:
    #     dtype = torch.float32

    # print(megatron_args)
    # # intermediate_size_map = {4096:11008,5120:13824,6656:17920,8192:22016,2560:6784}
    # config = LlamaConfig(
    #     vocab_size=vocab_size,
    #     hidden_size=megatron_args.hidden_size,
    #     num_hidden_layers=megatron_args.num_layers,
    #     num_attention_heads=megatron_args.num_attention_heads,
    #     num_key_value_heads=megatron_args.num_key_value_heads,
    #     intermediate_size=megatron_args.ffn_hidden_size,
    #     rms_norm_eps=1e-06,
    #     initializer_range=0.011,
    #     use_cache=True,
    #     pad_token_id=3,
    #     bos_token_id=1,
    #     eos_token_id=1,
    #     architectures=["LLaMAForCausalLM"],
    #     torch_dtype=dtype,
    #     max_sequence_length=2048,
    #     hidden_act="silu",
    # )

    # output_state_dict = {}

    # checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    # tp_size = megatron_args.tensor_model_parallel_size
    # pp_size = megatron_args.pipeline_model_parallel_size

    # # The regex to extract layer names.
    # layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # # Convert.
    # print("Converting")
    # print(tp_size, pp_size)

    # # Embeddings
    # print("Converting embeddings")
    # tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # # Convert and store the position embeddings.
    # position_embeddings = get_element_from_dict_by_path(
    #     tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    # )

    # if position_embeddings:
    #     output_state_dict["transformer.position_embeddings.weight"] = position_embeddings.to(dtype).clone()

    # # Convert and store the word embeddings.
    # word_embeddings = []
    # word_embeddings_layernorm_weight = []
    # word_embeddings_layernorm_bias = []

    # # import pdb
    # # pdb.set_trace()

    # for tp_rank in range(tp_size):
    #     embeddings = get_element_from_dict_by_path(
    #             tp_state_dicts[tp_rank], "model.word_embeddings_for_head.weight"
    #         )
    #     # After training with megatron, word_embeddings is stored differently
    #     if type(embeddings) is dict:
    #         embeddings = get_element_from_dict_by_path(
    #             tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
    #         )
    #     word_embeddings.append(embeddings)

    # word_embeddings = torch.cat(word_embeddings, dim=0)
    # word_embeddings = word_embeddings.to(dtype)
    # output_state_dict["model.embed_tokens.weight"] = word_embeddings.to(dtype).clone()
    # # Reset the vocab size
    # config.vocab_size = word_embeddings.shape[0]

    # # Transformer Layers
    # print("Converting transformer layers")
    # # The number of heads.
    # heads = config.num_attention_heads
    # # The hidden_size per head.
    # hidden_size_per_head = config.hidden_size // config.num_attention_heads
    # num_layers = config.num_hidden_layers // pp_size

    # for pp_rank in range(pp_size):
    #     if pp_size > 0:
    #         print(f"Converting pipeline parallel rank {pp_rank}")
    #         tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

    #     # The transformer.

    #     path = (
    #         "model.language_model.transformer"
    #         if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
    #         else "model.language_model.encoder"
    #     )

    #     # Extract the layers.
    #     for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
    #         # Match the name.
    #         m = layer_re.match(key)
    #         # Stop if that's not a layer
    #         if m is None:
    #             break

    #         # The index of the layer.
    #         layer_idx = int(m.group(1)) + pp_rank * num_layers
    #         # The name of the operation.
    #         op_name = m.group(2)
    #         # Is it a weight or a bias?
    #         weight_or_bias = m.group(3)

    #         # The name of the layer.
    #         layer_name = f"model.layers.{layer_idx}"

    #         if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
    #             params = val.to(dtype)
    #         else:
    #             # import pdb
    #             # pdb.set_trace()
    #             dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
    #             params = torch.cat(
    #                 [val]
    #                 + [
    #                     get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
    #                     for tp_rank in range(1, tp_size)
    #                 ],
    #                 dim=dim,
    #             ).to(dtype)

    #         # For layernorm(s), simply store the layer norm.
    #         if op_name.endswith("layernorm") and weight_or_bias == 'weight':
    #             ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
    #             output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params.clone()

    #         # Transpose the QKV matrix.
    #         elif (
    #             op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
    #         ) and weight_or_bias == "weight" and args.model_name != "llama2-70b":

    #             out_val = megatron_to_transformers_fix_query_key_value_ordering(
    #                 params,
    #                 checkpoint_version,
    #                 3,
    #                 heads,
    #                 hidden_size_per_head,
    #             )

    #             # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
    #             # out_val = out_val.transpose(0, 1).contiguous()
    #             # Store.

    #             # Split to QKV matrix
    #             QKV = {0:'q_proj',1:'k_proj',2:'v_proj'}
    #             for index, matrix in enumerate(torch.split(out_val, out_val.shape[1], 0)):
    #                 output_state_dict[layer_name + f".self_attn.{QKV[index]}.weight"] = matrix.clone()

    #         # Transpose the Q matrix for query for Llama70b.
    #         elif (
    #             op_name == "attention.query" or op_name == "self_attention.query"
    #         ) and weight_or_bias == "weight" and args.model_name == "llama2-70b":

    #             out_val = megatron_to_transformers_fix_query_key_value_ordering(
    #                 params,
    #                 checkpoint_version,
    #                 1,
    #                 heads,
    #                 hidden_size_per_head,
    #             )

    #             # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
    #             # out_val = out_val.transpose(0, 1).contiguous()
    #             # Store.

    #             # Split to QKV matrix
    #             output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_val.clone()

    #         # Transpose the KV matrix for query for Llama70b.
    #         elif (
    #             op_name == "attention.key_value" or op_name == "self_attention.key_value"
    #         ) and weight_or_bias == "weight" and args.model_name == "llama2-70b":

    #             out_val = megatron_to_transformers_fix_query_key_value_ordering(
    #                 params,
    #                 checkpoint_version,
    #                 2,
    #                 4,
    #                 hidden_size_per_head,
    #             )

    #             # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
    #             # out_val = out_val.transpose(0, 1).contiguous()
    #             # Store.

    #             # Split to QKV matrix
    #             KV = {0:'k_proj',1:'v_proj'}
    #             for index, matrix in enumerate(torch.split(out_val, out_val.shape[0]//2, 0)):
    #                 output_state_dict[layer_name + f".self_attn.{KV[index]}.weight"] = matrix.clone()

    #         # Transpose the weights.
    #         elif weight_or_bias == "weight":
    #             if 'dense_h_to_4h' in op_name:
    #                 out_name = megatron_to_transformers[op_name]
    #                 for index, mat in enumerate(torch.split(params, params.shape[0]//2)):
    #                     output_state_dict[layer_name + out_name[index] + "weight"] = mat.clone()
    #             else:
    #                 out_name = megatron_to_transformers[op_name]
    #                 output_state_dict[layer_name + out_name + "weight"] = params.clone()

    #         # Copy the bias.
    #         # Ignore them
    #         elif weight_or_bias == "bias":
    #             pass

    #         # Copy the Rotary Embedding
    #         else:
    #             out_name = megatron_to_transformers[op_name]
    #             output_state_dict[layer_name + out_name] = params.clone()

    # if config.num_hidden_layers != (layer_idx + 1):
    #     raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # # The final layernorm.
    # print("Converting final layernorm")
    # params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    # # print(params)
    # # print(str(path))
    # # print(tp_state_dicts)
    # output_state_dict["model.norm.weight"] = params["final_layernorm.weight"].to(dtype).clone() ### DEBUG

    # # For LM head, transformers' wants the matrix to weight embeddings.
    # print("Converting LM head")
    # params = torch.cat([
    #                     get_element_from_dict_by_path(tp_state_dicts[i], 'model.language_model.output_layer.weight')
    #                     for i in range(tp_size)]
    #     )
    
    # output_state_dict["lm_head.weight"] = params.to(dtype).clone()

    # # It should be done!
    # print("Conversion from Megatron-LM to Transformers is done!")

    # # Print the structure of converted state dict.
    # if args.print_checkpoint_structure:
    #     recursive_print(None, output_state_dict)

    # # Store the config to file.
    # print("Saving config")
    # config.save_pretrained(args.save_path)

    # # Store the state_dict to file.
    # max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    # shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # # Save the model
    # if not os.path.exists(args.save_path):
    #     os.system(f'mkdir -p {args.save_path}')
    # for shard_file, shard in shards.items():
    #     torch.save(shard, os.path.join(args.save_path, shard_file))

    # if index is None:
    #     print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    # else:
    #     save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
    #     # Save the index as well
    #     with open(save_index_file, "w", encoding="utf-8") as f:
    #         content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    #         f.write(content)
    #     print(
    #         f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
    #         f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
    #         f"index located at {save_index_file}."
    #     )
    
    # Save tokenizer 
    tokenizer = SEALIONTokenizer(megatron_args.tokenizer_model)
    tokenizer.save_pretrained(args.save_path)
    
    # modify tokenizer_config
    tokenizer_config_path = f'{args.save_path}/tokenizer_config.json'
    json_config = json.load(open(tokenizer_config_path))
    json_config['auto_map'] = {
        'AutoTokenizer': ["sealion_tokenization.SEALIONTokenizer", None]
    }
    with open(tokenizer_config_path, 'w') as f:
        json.dump(json_config, f, indent=2, sort_keys=True)
    shutil.copy('megatron/tokenizer/sealion_tokenization.py', args.save_path)

def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    # else:
    #     convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
