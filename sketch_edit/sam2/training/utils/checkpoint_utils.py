"""Minimal checkpoint utilities for loading UnSAMv2 models."""

import logging
from typing import Dict, Tuple


def split_lora_state_dict(state_dict: Dict) -> Tuple[Dict, Dict]:
    """Split state dict into base model weights and LoRA weights.

    UnSAMv2 checkpoints store LoRA weights with keys like:
    - layer.weight -> layer.original_layer.weight (base)
    - layer.lora_A, layer.lora_B (LoRA adapters)

    This function extracts the base weights and LoRA weights separately.

    Args:
        state_dict: Full state dictionary from checkpoint.

    Returns:
        Tuple of (base_state_dict, lora_state_dict).
    """
    base_state = {}
    lora_state = {}

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            # These are LoRA adapter weights
            lora_state[key] = value
        elif ".original_layer." in key:
            # Extract base weights from original_layer wrapper
            # e.g., "layer.original_layer.weight" -> "layer.weight"
            new_key = key.replace(".original_layer", "")
            base_state[new_key] = value
        else:
            # Regular weights (not part of LoRA layers)
            base_state[key] = value

    if lora_state:
        logging.info(f"Split checkpoint into {len(base_state)} base params and {len(lora_state)} LoRA params")
    else:
        logging.debug("No LoRA parameters found in checkpoint")

    return base_state, lora_state


def apply_lora_state_dict(model, lora_state_dict: Dict) -> list:
    """Apply LoRA weights to model.

    Args:
        model: The model to apply LoRA weights to.
        lora_state_dict: Dictionary of LoRA parameters.

    Returns:
        List of keys that were missing in the model.
    """
    if not lora_state_dict:
        return []

    model_state = dict(model.named_parameters())
    missing_keys = []

    for key, value in lora_state_dict.items():
        if key in model_state:
            model_state[key].data.copy_(value)
        else:
            missing_keys.append(key)

    if missing_keys:
        logging.warning(f"LoRA keys not found in model: {len(missing_keys)} keys")
    else:
        logging.info(f"Applied {len(lora_state_dict)} LoRA parameters successfully")

    return missing_keys
