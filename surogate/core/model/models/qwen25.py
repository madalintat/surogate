from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

"""
Instruct models:
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct

Base models:
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-7B
- Qwen/Qwen2.5-14B
- Qwen/Qwen2.5-32B
"""
register_model(
    ModelTemplate(
        LLMModelType.qwen2_5,
        ChatTemplateType.qwen2_5,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM']))


