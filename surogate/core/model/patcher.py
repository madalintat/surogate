import copy
from contextlib import contextmanager
from types import MethodType

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel

from surogate.core.model.model_info import ModelInfo
from surogate.core.model.registry import ModelTemplate
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()


def patch_get_input_embeddings(model, embedding_keys: str):
    def get_input_embeddings(self) -> nn.Module:
        return deep_getattr(model, embedding_keys)

    model.get_input_embeddings = MethodType(get_input_embeddings, model)


def get_lm_head_model(model, model_template: ModelTemplate = None, lm_heads=None):
    if isinstance(model, PeftModel):
        model = model.model

    return model


def patch_getattr(obj_cls, item_name: str):
    if hasattr(obj_cls, '_patch'):  # avoid double patch
        return

    def __new_getattr__(self, key: str):
        try:
            return super(self.__class__, self).__getattr__(key)
        except AttributeError:
            if item_name in dir(self):
                item = getattr(self, item_name)
                return getattr(item, key)
            raise

    obj_cls.__getattr__ = __new_getattr__
    obj_cls._patch = True

