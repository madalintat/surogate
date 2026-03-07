from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

"""
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3.5-2B
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-9B
- Qwen/Qwen3.5-27B
- Qwen/Qwen3.5-0.8B-Base
- Qwen/Qwen3.5-2B-Base
- Qwen/Qwen3.5-4B-Base
- Qwen/Qwen3.5-9B-Base
"""
register_model(
    ModelTemplate(
        model_type='Qwen3_5ForCausalLM',
        chat_templates=[ChatTemplateType.qwen3_5]))
