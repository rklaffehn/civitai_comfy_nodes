from .civitai_lora_loader import CivitAI_LORA_Loader
from .civitai_checkpoint_loader import CivitAI_Checkpoint_Loader
from .civitai_add_model_hash import CivitAI_AddModelHashes

NODE_CLASS_MAPPINGS = {
    "CivitAI_Lora_Loader": CivitAI_LORA_Loader,
    "CivitAI_Checkpoint_Loader": CivitAI_Checkpoint_Loader,
    "CivitAI_AddModelHashes": CivitAI_AddModelHashes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitAI_Lora_Loader": "CivitAI Lora Loader",
    "CivitAI_Checkpoint_Loader": "CivitAI Checkpoint Loader",
    "CivitAI_AddModelHashes": "CivitAI Add Model Hashes"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']