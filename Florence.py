import os
import io
import copy
import gc
from unittest.mock import patch
import random

from PIL import Image, ImageDraw, ImageFont 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Workaround for FlashAttention"""
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

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return Image.open(buf)

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(image.width / 100, image.height / 100)
    ax.imshow(image)
    for i, (bbox, label) in enumerate(zip(data['bboxes'], data['labels'])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        enum_label = f"{i}: {label}"
        plt.text(x1 + 7, y1 + 17, enum_label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = color if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return output_image

def convert_to_od_format(data):
    od_results = {
        'bboxes': data.get('bboxes', []),
        'labels': data.get('bboxes_labels', [])
    }
    return od_results

def draw_ocr_bboxes(image, prediction):
    scale = 1
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return output_image

TASK_OPTIONS = [
    "caption",
    "detailed caption",
    "more detailed caption",
    "object detection",
    "dense region caption",
    "region proposal",
    "caption to phrase grounding",
    "referring expression segmentation",
    "region to segmentation",
    "open vocabulary detection",
    "region to category",
    "region to description",
    "OCR",
    "OCR with region"
    ]

class Florence2:
    def __init__(self):
        self.model = None
        self.processor = None
        self.version = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "version": (["base", "base-ft", "large", "large-ft"],),
                "task": (TASK_OPTIONS, {"default": TASK_OPTIONS[0]}),
                "text_input": ("STRING", {}),
                "keep_loaded": ("BOOLEAN", {"default": True,}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "F_BBOXES",)
    RETURN_NAMES = ("preview", "string", "F_BBOXES",)
    FUNCTION = "apply"
    CATEGORY = "Florence2"
    
    def apply(self, image, version, task, text_input, keep_loaded):
        img = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)) 
        
        if self.version != version:
            self.model, self.processor = load_model(version, self.device)
            self.version = version
        
        results, output_image = self.process_image(img, task, text_input)
        # bboxes, labels OR polygons, labels
        if isinstance(results, dict):
            results["width"] = img.width
            results["height"] = img.height
        
        if output_image == None:
            output_image = image[0].detach().clone().unsqueeze(0)
        else:
            output_image = np.asarray(output_image).astype(np.float32) / 255
            output_image = torch.from_numpy(output_image).unsqueeze(0)
        
        if not keep_loaded:
            self.model = None
            self.processor = None
            self.version = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return (output_image, str(results), results)
    
    def run_example(self, task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer
    
    def process_image(self, image, task_prompt, text_input=None):
        if task_prompt == 'caption':
            task_prompt = '<CAPTION>'
            result = self.run_example(task_prompt, image)
            return result[task_prompt], None
        elif task_prompt == 'detailed caption':
            task_prompt = '<DETAILED_CAPTION>'
            result = self.run_example(task_prompt, image)
            return result[task_prompt], None
        elif task_prompt == 'more detailed caption':
            task_prompt = '<MORE_DETAILED_CAPTION>'
            result = self.run_example(task_prompt, image)
            return result[task_prompt], None
        elif task_prompt == 'object detection':
            task_prompt = '<OD>'
            results = self.run_example(task_prompt, image)
            fig = plot_bbox(image, results['<OD>'])
            return results[task_prompt], fig_to_pil(fig)
        elif task_prompt == 'dense region caption':
            task_prompt = '<DENSE_REGION_CAPTION>'
            results = self.run_example(task_prompt, image)
            fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
            return results[task_prompt], fig_to_pil(fig)
        elif task_prompt == 'region proposal':
            task_prompt = '<REGION_PROPOSAL>'
            results = self.run_example(task_prompt, image)
            fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
            return results[task_prompt], fig_to_pil(fig)
        elif task_prompt == 'caption to phrase grounding':
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            results = self.run_example(task_prompt, image, text_input)
            fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
            return results[task_prompt], fig_to_pil(fig)
        elif task_prompt == 'referring expression segmentation':
            task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
            results = self.run_example(task_prompt, image, text_input)
            output_image = draw_polygons(image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
            return results[task_prompt], output_image
        elif task_prompt == 'region to segmentation':
            task_prompt = '<REGION_TO_SEGMENTATION>'
            results = self.run_example(task_prompt, image, text_input)
            output_image = draw_polygons(image, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)
            return results[task_prompt], output_image
        elif task_prompt == 'open vocabulary detection':
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            results = self.run_example(task_prompt, image, text_input)
            bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
            fig = plot_bbox(image, bbox_results)
            return bbox_results, fig_to_pil(fig)
        elif task_prompt == 'region to category':
            task_prompt = '<REGION_TO_CATEGORY>'
            results = self.run_example(task_prompt, image, text_input)
            return results[task_prompt], None
        elif task_prompt == 'region to description':
            task_prompt = '<REGION_TO_DESCRIPTION>'
            results = self.run_example(task_prompt, image, text_input)
            return results[task_prompt], None
        elif task_prompt == 'OCR':
            task_prompt = '<OCR>'
            result = self.run_example(task_prompt, image)
            return result[task_prompt], None
        elif task_prompt == 'OCR with region':
            task_prompt = '<OCR_WITH_REGION>'
            results = self.run_example(task_prompt, image)
            output_image = draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])
            output_results = {'bboxes': results[task_prompt].get('quad_boxes', []),
                              'labels': results[task_prompt].get('labels', [])}
            return output_results, output_image
        else:
            return "", None  # Return empty string and None for unknown task prompts

class Florence2Postprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "F_BBOXES": ("F_BBOXES",),
                "index": ("INT", {"default": 0, "min": 0}),
            },
        }
    
    RETURN_TYPES = ("MASK", "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "label", "loc_string", "width", "height", "x", "y")
    FUNCTION = "apply"
    CATEGORY = "Florence2"
    
    def apply(self, F_BBOXES, index):
        if isinstance(F_BBOXES, str):
            return (torch.zeros(1, 512, 512, dtype=torch.float32), F_BBOXES)
        
        width = F_BBOXES["width"]
        height = F_BBOXES["height"]
        mask = np.zeros((height, width), dtype=np.uint8)

        x1 = y1 = x2 = y2 = 0
        label = ""
        if index < len(F_BBOXES["labels"]):
            if "bboxes" in F_BBOXES:
                bbox = F_BBOXES["bboxes"][index]
                label = F_BBOXES["labels"][index]
                label = label.removeprefix("</s>")

                if len(bbox) == 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                elif len(bbox) == 8:
                    x1 = int(min(bbox[0::2]))
                    x2 = int(max(bbox[0::2]))
                    y1 = int(min(bbox[1::2]))
                    y2 = int(max(bbox[1::2]))

                mask[y1:y2, x1:x2] = 1

            else:
                polygon = F_BBOXES["polygons"][0][0]
                label = F_BBOXES["labels"][index]

                image = Image.new('RGB', (width, height), color='black')
                draw = ImageDraw.Draw(image)
                _polygon = np.array(polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    print('Invalid polygon:', _polygon)
                else:
                    _polygon = (_polygon).reshape(-1).tolist()
                    draw.polygon(_polygon, outline='white', fill='white')

                x1 = int(min(polygon[0::2]))
                x2 = int(max(polygon[0::2]))
                y1 = int(min(polygon[1::2]))
                y2 = int(max(polygon[1::2]))

                mask = np.asarray(image)[..., 0].astype(np.float32) / 255
        
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        loc_string = f"<loc_{x1 * 999 // width}><loc_{y1 * 999 // height}><loc_{x2 * 999 // width}><loc_{y2 * 999 // height}>"
        return (mask, label, loc_string, x2 - x1 + 1, y2 - y1 + 1, x1, y1)

class Florence2PostprocessMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "F_BBOXES": ("F_BBOXES",),
            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "apply"
    CATEGORY = "Florence2"

    def apply(self, F_BBOXES):
        if isinstance(F_BBOXES, str):
            return (torch.zeros(1, 512, 512, dtype=torch.float32), F_BBOXES)

        width = F_BBOXES["width"]
        height = F_BBOXES["height"]
        mask = np.zeros((height, width), dtype=np.uint8)

        if "bboxes" in F_BBOXES:
            for idx in range(len(F_BBOXES["bboxes"])):
                bbox = F_BBOXES["bboxes"][idx]

                if len(bbox) == 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                elif len(bbox) == 8:
                    x1 = int(min(bbox[0::2]))
                    x2 = int(max(bbox[0::2]))
                    y1 = int(min(bbox[1::2]))
                    y2 = int(max(bbox[1::2]))
                else:
                    continue

                mask[y1:y2, x1:x2] = 1

        else:
            polygon = F_BBOXES["polygons"][0][0]

            image = Image.new('RGB', (width, height), color='black')
            draw = ImageDraw.Draw(image)
            _polygon = np.array(polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
            else:
                _polygon = (_polygon).reshape(-1).tolist()
                draw.polygon(_polygon, outline='white', fill='white')

            # x1 = int(min(polygon[0::2]))
            # x2 = int(max(polygon[0::2]))
            # y1 = int(min(polygon[1::2]))
            # y2 = int(max(polygon[1::2]))

            mask = np.asarray(image)[..., 0].astype(np.float32) / 255

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return (mask, )

NODE_CLASS_MAPPINGS = {
    "Florence2": Florence2,
    "Florence2Postprocess": Florence2Postprocess,
    "Florence2PostprocessMasks": Florence2PostprocessMasks,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2": "Florence2",
    "Florence2Postprocess": "Florence2 Postprocess",
    "Florence2PostprocessMasks": "Florence2 Postprocess Masks",
    }