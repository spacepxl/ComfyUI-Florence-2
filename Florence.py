import os
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/Florence-2-large-ft/discussions/4"""
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def load_model(version, device):
    model_dir = os.path.join(os.path.split(__file__)[0], "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    identifier = "microsoft/Florence-2-" + version
    
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(identifier, cache_dir=model_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(identifier, cache_dir=model_dir, trust_remote_code=True)
    
    model = model.to(device)
    return (model, processor)

class Florence2:
    def __init__(self):
        self.model = None
        self.processor = None
        self.version = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "version": (["base-ft", "large-ft"],),
                "task": (["caption", "detailed caption", "more detailed caption"],),
                "keep_loaded": ("BOOLEAN", {"default": True,}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    # RETURN_NAMES = ("",)
    FUNCTION = "apply"
    CATEGORY = "Florence2"

    def apply(self, image, version, task, keep_loaded):
        img = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.version == version:
            model = self.model
            processor = self.processor
        else:
            model, processor = load_model(version, device)
        
        task_prompt = "<CAPTION>"
        if task == "detailed caption":
            task_prompt = "<DETAILED_CAPTION>"
        elif task == "more detailed caption":
            task_prompt = "<MORE_DETAILED_CAPTION>"
        
        inputs = processor(text=task_prompt, images=img, return_tensors="pt")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            max_new_tokens=1024,
            num_beams=3
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(img.width, img.height))
        
        if keep_loaded:
            self.model = model
            self.processor = processor
            self.version = version
        else:
            self.model = None
            self.processor = None
            self.version = None
            del model, processor
        
        return (image[0].detach().clone().unsqueeze(0), parsed_answer[task_prompt])


NODE_CLASS_MAPPINGS = {
    "Florence2": Florence2,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2": "Florence 2",
    }