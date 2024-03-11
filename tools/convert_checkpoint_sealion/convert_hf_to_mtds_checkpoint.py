import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepspeed
import torch
from deepspeed.runtime.utils import see_memory_usage
from megatron import print_rank_0, get_args
from megatron.arguments import core_transformer_config_from_args
from megatron.checkpointing import save_checkpoint
from megatron.core import mpu
from megatron.core.utils import divide
from megatron.initialize import initialize_megatron
from megatron.model import GPTModelPipe, Float16Module
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.optimizer import get_megatron_optimizer
from megatron.training import get_optimizer_param_scheduler
from megatron.utils import unwrap_model
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


def add_extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='mt to mtds')
    group.add_argument("--origin-mt-ckpt-dir", type=str, default="",
                       help="Path to the converted MT checkpoint.")
    return parser


def compute_partition_range(hidden_size, local_rank, tp_size):
    partition_size = divide(hidden_size, tp_size)
    start_index = local_rank * partition_size
    end_index = start_index + partition_size
    return partition_size, start_index, end_index


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


def load_and_print_mt_weight(mt_ckpt_dir):
    print_rank_0("----------------------------mt weight list----------------------------")

    state_dict = torch.load(
        f"{mt_ckpt_dir}/mp_rank_00/model_optim_rng.pt", map_location=torch.device("cpu")
    )
    recursive_print(None, state_dict)
    return state_dict


def print_distinct_weights(model):
    print_rank_0("----------------------------mtds weight list----------------------------")
    for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):
        if mpu.get_pipeline_model_parallel_rank() == pipe_rank:
            if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                for pname, p in model.named_parameters():
                    print(pname)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()


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
            print_rank_0(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print_rank_0(f"{msg} : {val.size()}")
    else:
        print_rank_0(f"{msg} : {val}")


class Refactor:
    def __init__(self, model, loaded, args, config):
        # align layer number
        self.model = model
        self.loaded = loaded
        self.config = config

        self.idx_offset = 2
        self.mtds_embedding_idx = 1
        self.mtds_final_layernorm_idx = self.mtds_embedding_idx + args.num_layers + 1
        self.mtds_lm_head_idx = self.mtds_final_layernorm_idx + 1

        self.padded_vocab_size = args.padded_vocab_size
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.decoder_re = re.compile("(\d+)\.(.+)")
        self.refactor_weight_list = []
        self.is_refactored = False

    def refactor(self):
        assert self.is_refactored == False
        new_w = None
        for param_name, params in self.model.named_parameters():
            if param_name in [
                    f"{self.mtds_embedding_idx}.word_embeddings.weight",
                    f"{self.mtds_lm_head_idx}.lm_head.weight",
            ]:
                new_w = self._refactor_embedding(param_name, params)
            elif param_name == f"{self.mtds_final_layernorm_idx}.weight":
                new_w = self._direct_refactor(param_name, params)
            else:
                m = self.decoder_re.match(param_name)
                # Index of the layer
                layer_idx = int(m.group(1))
                # <Name of the operation>.weight
                op_name = m.group(2)
                mt_layer_idx = layer_idx - self.idx_offset
                # ColumnParallelLinear layers
                if op_name in ["self_attention.query.weight", "self_attention.key_value.weight"]:
                    new_w = self._refactor_qkv(param_name, params, mt_layer_idx, op_name)
                # ColumnParallelLinear layers
                elif op_name in ["mlp.dense_h_to_4h.weight"]:
                    new_w = self._refactor_dense_h_to_4h(param_name, params, mt_layer_idx, op_name)
                # RowParallelLinear layers
                elif op_name in ["self_attention.dense.weight", "mlp.dense_4h_to_h.weight"]:
                    new_w = self._refactor_row_parallel(param_name, params, mt_layer_idx, op_name)
                elif op_name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                    new_w = self._direct_refactor(param_name, params, mt_layer_idx, op_name)
                else:
                    raise ValueError("Unrecognized weight type")
            params.data.copy_(new_w)
            new_w = None
        self.is_refactored = True

    def _direct_refactor(self, param_name, params, layer_idx=None, op_name=None):
        if param_name == f"{self.mtds_final_layernorm_idx}.weight":
            mt_name = "final_layernorm.weight"
        elif op_name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            assert layer_idx is not None and op_name is not None
            mt_name = f"layers.{layer_idx}.{op_name}"
        mt_wei = get_element_from_dict_by_path(self.loaded, "model.language_model.encoder")[mt_name]

        new_w = mt_wei
        self.record_mapping_info(f"mtds: {param_name,params.data.shape}<--mt: {mt_name,} {mt_wei.shape}")
        return new_w

    def _refactor_embedding(self, param_name, params):
        if param_name == f"{self.mtds_lm_head_idx}.lm_head.weight":
            mt_name = "model.language_model.output_layer.weight"
        elif param_name == f"{self.mtds_embedding_idx}.word_embeddings.weight":
            mt_name = "model.language_model.embedding.weight"
        mt_wei = get_element_from_dict_by_path(self.loaded, mt_name)
        assert mt_wei.shape[0] == self.padded_vocab_size

        per_partition_vocab_size, start_index, end_index = compute_partition_range(
            self.padded_vocab_size, self.tp_rank, self.tp_size)

        new_w = torch.zeros((per_partition_vocab_size, mt_wei.shape[1]), dtype=mt_wei.dtype)
        new_w[:per_partition_vocab_size, :] = mt_wei[start_index:end_index, :]

        self.record_mapping_info(
            f"mtds: {param_name,params.data.shape}<--mt: {mt_name,} [{start_index}:{end_index}, :] of {mt_wei.shape}"
        )
        return new_w

    def _refactor_qkv(self, param_name, params, layer_idx, op_name):
        if op_name == "self_attention.query.weight":
            num_heads = self.config.num_attention_heads
        elif op_name == "self_attention.key_value.weight":
            num_heads = self.config.num_key_value_heads
        mt_name = f"layers.{layer_idx}.{op_name}"
        mt_wei = get_element_from_dict_by_path(self.loaded, "model.language_model.encoder")[mt_name]

        projection_size = mt_wei.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            projection_size, self.tp_rank, self.tp_size
        )
        head_dim = divide(projection_size, num_heads)
        num_heads_per_partition = divide(num_heads, self.tp_size)

        new_w = torch.zeros((per_partition_size, mt_wei.shape[1]), dtype=mt_wei.dtype)
        new_w[:per_partition_size, :] = mt_wei[start_index:end_index, :]

        self.record_mapping_info(
            f"mtds: {param_name,params.data.shape}<--mt: {mt_name,} [{start_index}:{end_index}, :] of {mt_wei.shape}"
        )
        return new_w

    def _refactor_dense_h_to_4h(self, param_name, params, layer_idx, op_name):
        mt_name = f"layers.{layer_idx}.{op_name}"
        mt_wei = get_element_from_dict_by_path(self.loaded, "model.language_model.encoder")[mt_name]

        ffn_hidden_size = mt_wei.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            ffn_hidden_size, self.tp_rank, self.tp_size
        )
        new_w = torch.zeros((per_partition_size, mt_wei.shape[1]), dtype=mt_wei.dtype)
        new_w[:per_partition_size, :] = mt_wei[start_index:end_index, :]

        self.record_mapping_info(
            f"mtds: {param_name,params.data.shape}<--mt: {mt_name} [{start_index}:{end_index},:] of {mt_wei.shape}"
        )
        return new_w

    def _refactor_row_parallel(self, param_name, params, layer_idx, op_name):
        mt_name = f"layers.{layer_idx}.{op_name}"
        mt_wei = get_element_from_dict_by_path(self.loaded, "model.language_model.encoder")[mt_name]

        hidden_size = mt_wei.shape[1]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size
        )
        new_w = torch.zeros((mt_wei.shape[0], per_partition_size), dtype=mt_wei.dtype)
        new_w[:, :per_partition_size] = mt_wei[:, start_index:end_index]
        self.record_mapping_info(
            f"mtds: {param_name,params.data.shape}<--mt: {mt_name,}  [:,{start_index}:{end_index}] of {mt_wei.shape}"
        )
        return new_w


    def record_mapping_info(self, record_msg):
        self.refactor_weight_list.append(record_msg)

    def inorder_show_record(self):
        assert self.is_refactored
        print_rank_0("----------------------------mapping list----------------------------")
        # print dp rank0 tp rank0  records.
        for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):
            if mpu.get_pipeline_model_parallel_rank() == pipe_rank:
                if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                    for record in self.refactor_weight_list:
                        print(record)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()


def convert_hf_to_mega_ds():
    """Build the model."""
    args = get_args()
    print_rank_0("building model ...")
    see_memory_usage("Before Building Model", force=True)

    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(
        data_parallel_group=mpu.get_data_parallel_group(),
        remote_device=None if args.remote_device == "none" else args.remote_device,
        config_dict_or_path=args.deepspeed_config,
        enabled=args.zero_stage == 3,
        mpu=mpu
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(config, num_tokentypes=0, parallel_output=True)
        else:
            raise NotImplementedError("Not implemented")

    see_memory_usage("After Building Model", force=True)
    if torch.distributed.get_rank() < 2:
        print(f"{torch.distributed.get_rank()} {model}")

    # load and initialize HF weight dict
    # print hf weights list & mega-ds weights list
    mt_ckpt_dir = args.origin_mt_ckpt_dir
    loaded = load_and_print_mt_weight(mt_ckpt_dir)
    print_distinct_weights(model)

    # refactor weight from mt to mtds
    cur_refactor = Refactor(model, loaded, args, config)
    cur_refactor.refactor()
    cur_refactor.inorder_show_record()

    del loaded

    unwrapped_model = unwrap_model([model], (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    #init model and save
    print_rank_0(f"before deepspeed init")
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None)
    print_rank_0(f"after deepspeed init")

    print_rank_0(f"mega-ds checkpoint will be saved in {args.save}")
    save_checkpoint(0, [ds_engine], optimizer, opt_param_scheduler)
    print_rank_0(f"save checkpoint completed")


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_extra_args)
    convert_hf_to_mega_ds()

