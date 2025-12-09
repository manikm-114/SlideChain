# compare_idefics2.py
import os
import re
import json
import warnings
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

MODELS = [
    "HuggingFaceM4/idefics2-8b",
]

PROMPTS: Dict[str, str] = {
    "concepts": """System:
You are an expert educator in medical imaging. From the following slide text,
extract the key technical or scientific concepts that are essential for understanding the topic.

User:
SLIDE_TEXT:
<<SLIDE_TEXT>>

STRICT INSTRUCTIONS:
- Return pure JSON only. No prose, no markdown, no code fences.
- Each concept must appear verbatim (case-insensitive) in SLIDE_TEXT.
- Assign exactly one category from:
  software | workflow | mathematics | signal_processing | frequency_domain |
  physics | instrumentation | data_processing | reconstruction |
  quality_metric | communication | modality | anatomy | algorithm | ai_ml
- Reject filler or administrative text.
- Output format:
{
  "concepts": [
    {"term": "<exact term from slide>", "category": "<category>"}
  ],
  "evidence": ["<short supporting phrase from SLIDE_TEXT>"]
}
""",
    "triples": """System:
You are extracting factual relations from a medical imaging lecture. Use information grounded in the input SLIDE_TEXT.

User:
SLIDE_TEXT:
<<SLIDE_TEXT>>

STRICT INSTRUCTIONS:
- Return JSON only. No prose, no markdown, no code fences.
- Output a triple only if subject and object appear verbatim in SLIDE_TEXT (case-insensitive).
- Predicates: uses | via | represents | depends_on | measures | produces | reconstructs_with.
- Add modalities ["text"]; add "image" only if visible without text.
- Confidence in [0,1].

OUTPUT:
{
  "triples": [
    {"s": "<verbatim subject>", "p": "<predicate>", "o": "<verbatim object>",
     "modalities": ["text","image"], "confidence": 0.0, "evidence": "<short quote>"}
  ]
}
""",
}

GEN_KW = dict(max_new_tokens=256, do_sample=False, use_cache=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT, "Images")
TEXT_DIR = os.path.join(ROOT, "Texts")
OUT_DIR = os.path.join(ROOT, "Outputs")
warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_text(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def list_slides(dir_path: str) -> List[str]:
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    return files

class Idefics2Model:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        # Do NOT call self.model.to(device) because device_map="auto" handles this

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt_text: str, gen_kw: Dict[str, Any]) -> str:
        input_text = f"<image>\n{prompt_text}"
        inputs = self.processor(
            text=[input_text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kw_sanitized = {k: v for k, v in gen_kw.items() if k in self.model.generation_config.to_dict()}
        output_ids = self.model.generate(**inputs, **gen_kw_sanitized)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output_text.strip()

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    slides = list_slides(IMAGE_DIR)
    if not slides:
        raise FileNotFoundError(f"No slides found in {IMAGE_DIR}")

    for model_id in MODELS:
        short_name = model_id.replace("/", "__")
        try:
            model_runner = Idefics2Model(model_id, device)
            print(f"✅ Loaded model: {model_id}")
        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {e}")
            continue

        for prompt_id, prompt_template in PROMPTS.items():
            out_dir = os.path.join(OUT_DIR, short_name, prompt_id)
            ensure_dir(out_dir)
            print(f"\n=== Running model={model_id} prompt={prompt_id} on {len(slides)} slides ===")

            success_count = 0
            for slide_file in tqdm(slides, desc=f"{short_name} | {prompt_id}"):
                slide_id = os.path.splitext(slide_file)[0]
                img_path = os.path.join(IMAGE_DIR, slide_file)
                txt_path = os.path.join(TEXT_DIR, f"{slide_id}.txt")

                if not os.path.isfile(txt_path):
                    print(f"⚠️ Missing text for slide {slide_id}, skipping.")
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"⚠️ Failed to open image {img_path}: {e}")
                    continue

                slide_text = read_text(txt_path)
                prompt_text = prompt_template.replace("<<SLIDE_TEXT>>", slide_text)

                raw_output = model_runner.generate(image=image, prompt_text=prompt_text, gen_kw=GEN_KW)
                try:
                    parsed = json.loads(raw_output) if raw_output.strip().startswith("{") else None
                except:
                    parsed = None

                if parsed:
                    success_count += 1
                    # Optional: filtering or validation here

                record = {
                    "slide_id": slide_id,
                    "model": model_id,
                    "prompt": prompt_id,
                    "raw_output": raw_output,
                    "parsed": parsed
                }

                with open(os.path.join(out_dir, f"{slide_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"✅ Completed {success_count}/{len(slides)} slides for prompt '{prompt_id}'.")

if __name__ == "__main__":
    run()
