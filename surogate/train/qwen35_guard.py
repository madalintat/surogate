import os


def apply_qwen35_sample_packing_guard(config, logger) -> bool:
    """Disable sample packing by default for Qwen3.5 training.

    Returns True when config.sample_packing was changed from True -> False.
    """
    model_info = getattr(config, "model_info", None)
    template_type = str(getattr(model_info, "template_type", "") or "")
    if not template_type.startswith("Qwen3_5"):
        return False
    if not getattr(config, "sample_packing", False):
        return False

    if os.getenv("SUROGATE_ALLOW_QWEN35_SAMPLE_PACKING", "0") == "1":
        logger.warning(
            "Qwen3.5 with sample_packing=true is experimental and may be numerically unstable."
        )
        return False

    logger.warning(
        "Qwen3.5 disables sample_packing by default (known instability with packed samples); "
        "forcing sample_packing=False. Set SUROGATE_ALLOW_QWEN35_SAMPLE_PACKING=1 to override."
    )
    config.sample_packing = False
    return True
