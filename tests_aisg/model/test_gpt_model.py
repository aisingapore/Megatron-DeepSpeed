import sys

import pytest

from megatron import initialize_megatron
from megatron.core.enums import ModelType
from megatron.training import pretrain
from pretrain_gpt import (
    data_post_process,
    forward_step,
    model_provider,
    train_valid_test_datasets_provider,
)


class TestGPTModel:
    def test_pretrain(self, mock_args):
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            data_post_process=data_post_process,
        )

