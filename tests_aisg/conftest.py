import os
import shutil
import sys
from pathlib import Path

import pytest
import torch.distributed


CWD = Path(__file__).resolve().parent


@pytest.fixture
def mock_args():
    workspace_dir = Path(
        f"/scratch/users/nus/{os.environ['USER']}/ds_megatron_test_output"
    )
    shared_fs_dir = Path(f"{os.environ['AISG_ENG_DIR']}/multi-node")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = f"{workspace_dir}/checkpoint/gpt_0.125B_R1"
    tensorboard_dir = f"{workspace_dir}/tensorboard"
    tokenizer_path = f"{shared_fs_dir}/tokenizer/256000_wonorm_wdummyprefix.model"
    data_path = (
        f"{shared_fs_dir}/data/enwiki/hfbpe_gpt_training_data_text_document"
    )
    config_path = _generate_deepspeed_config(workspace_dir)

    orig_argv = sys.argv.copy()
    sys.argv = sys.argv[:1] + [
        '--override-opt_param-scheduler',
        '--adam-beta1', '0.9',
        '--adam-beta2', '0.95',
        '--tensor-model-parallel-size', '2',
        '--moe-expert-parallel-size', '4',
        '--num-experts', '8',
        '--moe-loss-coeff', '0.01',
        '--moe-train-capacity-factor', '1.0',
        '--moe-eval-capacity-factor', '1.0',
        '--moe-min-capacity', '4',
        '--init-method-std', '0.014',
        '--lr-decay-tokens', '300000000000',
        '--lr-warmup-tokens', '375000000',
        '--micro-batch-size', '4',
        '--exit-duration-in-mins', '30000000',
        '--global-batch-size', '16',
        '--num-layers', '4',
        '--hidden-size', '12',
        '--num-attention-heads', '4',
        '--seq-length', '2048',
        '--max-position-embeddings', '2048',
        '--train-tokens', '300000000000',
        '--train-iters', '1',
        '--lr', '4.5e-4',
        '--min-lr', '4.5e-06',
        '--lr-decay-style', 'cosine',
        '--split', '98,2,0',
        '--log-interval', '1',
        '--eval-interval', '100',
        '--eval-iters', '0',
        '--save-interval', '1',
        '--weight-decay', '0.1',
        '--clip-grad', '1.0',
        '--hysteresis', '2',
        '--num-workers', '0',
        '--bf16',
        '--load', str(checkpoint_dir),
        '--save', str(checkpoint_dir),
        '--tensorboard-queue-size', '1',
        '--log-timers-to-tensorboard',
        '--log-batch-size-to-tensorboard',
        '--log-validation-ppl-to-tensorboard',
        '--tensorboard-dir', str(tensorboard_dir),
        '--checkpoint-activations',
        '--create-moe-param-group',
        '--tokenizer-type', 'SentencePieceTokenizer',
        '--tokenizer-model', str(tokenizer_path),
        '--data-path', str(data_path),
        '--data-impl', 'mmap',
        '--deepspeed',
        '--deepspeed_config', str(config_path),
        '--pipeline-model-parallel-size', '1',
        '--no-pipeline-parallel',
        '--deepspeed-activation-checkpointing',
    ]

    yield

    sys.argv = orig_argv
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(workspace_dir)

def _generate_deepspeed_config(parent_dir):
    dst = parent_dir / "ds_config.json"
    shutil.copyfile(CWD / "ds_config.json", dst)
    return dst


