import folder_paths
import os

from .CivitAI_Model import CivitAI_Model

class CivitAI_AddModelHashes:
    '''Parse the prompt (image meta-data) and add model hashes for various models.
       We're looking for input widgets with specific names and will examine the model
       files to generate the hashes.'''

    @classmethod
    def INPUT_TYPES(self):
        return {"required": {"image": ("IMAGE",)},
                "hidden": {"prompt": "PROMPT"}
                }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "execute"

    OUTPUT_NODE = True
    CATEGORY = "CivitAI/Tools"

    def execute(self, image, prompt: dict = None):
        for id, node in prompt.items():
            try:
                self._visitNode(node)
            except:
                pass

        return (image, {"prompt": prompt})

    def _visitNode(self, node: dict) -> dict:
        MODEL_PARAMETERS = {"ckpt_name": "checkpoints",
                            "lora_name": "loras", "vae_name": "vae"}

        if not "inputs" in node or not isinstance(node["inputs"], dict):
            return

        for parameter, value in node["inputs"].items():
            if parameter in MODEL_PARAMETERS.keys():
                model_path = os.path.normpath(
                    folder_paths.get_full_path(MODEL_PARAMETERS[parameter], value))

                digest = CivitAI_Model.calculate_sha256(model_path)
                if digest:
                    hash_key = str(parameter).replace("_name", "_hash")
                    node.update({hash_key: digest[:10]})
