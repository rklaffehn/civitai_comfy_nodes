import folder_paths
import os
import re

from .CivitAI_Model import CivitAI_Model

import pprint


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
        nodes = prompt.values()
        containers = list(filter(lambda node: str(
            node["class_type"]) == 'CivitAI_AddModelHashes', nodes))
        if not containers:
            containers = [dict()]

        for container in containers:
            if "hashes" in container:
                del container["hashes"]

        for node in nodes:
            try:
                self._visitNode(containers[0], node)
            except Exception as e:
                pprint.pprint(e)
                pass

        pprint.pprint(containers[0])

        return (image, {"prompt": prompt})

    def _visitNode(self, container: dict, node: dict) -> dict:
        if not "inputs" in node or not isinstance(node["inputs"], dict):
            return

        inputs = node["inputs"]

        if str(node["class_type"]).startswith('CLIPTextEncode'):
            self._addEmbeddingHashes(container, node, inputs)
        else:
            self._addModelHashes(container, node, inputs)

    def _addModelHashes(self, container: dict, node: dict, inputs: dict):
        MODEL_PARAMETERS = {"ckpt_name": ("checkpoints", lambda name, hash: {'model': hash}),
                            "lora_name": ("loras", lambda name, hash: {f'lora:{os.path.splitext(os.path.basename(name))[0]}': hash}),
                            "vae_name": ("vae", lambda name, hash: {'vae': hash})}

        for parameter, value in inputs.items():
            if parameter in MODEL_PARAMETERS.keys():
                model_path = os.path.normpath(
                    folder_paths.get_full_path(MODEL_PARAMETERS[parameter][0], value))

                digest = CivitAI_Model.calculate_sha256(model_path)
                if digest:
                    hash_key = str(parameter).replace("_name", "_hash")
                    node.update({hash_key: digest[:10]})

                    hashes = MODEL_PARAMETERS[parameter][1](
                        os.path.basename(model_path), digest[:10])
                    self._addHashesToContainer(container, hashes)

    def _addEmbeddingHashes(self, container: dict, node: dict, inputs: dict):
        content = ""
        for key in [key for key in inputs.keys() if key.startswith('text')]:
            content = content + '\n' + str(inputs[key])

        matcher = re.compile(
            r'embedding:([a-zA-Z0-9_\-\\/]+(?:\.[a-zA-Z0-9]+)?)')

        available = folder_paths.get_filename_list('embeddings')

        embeddings = dict()
        for match in matcher.finditer(content):
            embedding = os.path.normpath(match.group(1))
            candidates = [os.path.normpath(c) for c in filter(
                lambda e: os.path.normpath(e) == embedding or os.path.basename(e).startswith(embedding), available)]
            if candidates:
                embedding = candidates[0]
                model_path = os.path.normpath(
                    folder_paths.get_full_path('embeddings', embedding))

                digest = CivitAI_Model.calculate_sha256(model_path)
                if digest:
                    embeddings[f'embed:{os.path.splitext(os.path.basename(embedding))[0]}'] = digest[:10]

            if len(embeddings):
                node.update({"embedding_hashes": embeddings})
                self._addHashesToContainer(container, embeddings)

    def _addHashesToContainer(self, container, hashes):
        if not "hashes" in container:
            container["hashes"] = hashes
        else:
            container["hashes"].update(hashes)
