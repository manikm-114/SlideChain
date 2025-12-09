# compare_internvl.py
import os
import re
import json
import warnings
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, GenerationConfig

MODELS = [
    #"OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL3-8B", #use only this one
    #"OpenGVLab/InternVL3_5-8B",
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
- Category guidance:
  * software → programming tools/environments (MATLAB, Octave, MathWorks)
  * workflow → procedural/course steps (BUT exclude admin like office hours)
  * mathematics / signal_processing / frequency_domain → formulas, transforms, filters, spectra
  * physics → energy, radiation, waves, gradients, attenuation
  * instrumentation → scanners, detectors, coils, transducers, gantry
  * data_processing / reconstruction → corrections, backprojection, iterative, FBP, FFT usage
  * quality_metric → SNR, resolution, artifacts, MTF, DQE
  * communication → network or data transfer
  * modality → CT, MRI, PET, SPECT, Ultrasound (US), Optical/Photoacoustic, etc.
  * anatomy → biological structures (organs, tissues, bones, vessels) ONLY
  * algorithm → analytical/iterative computational methods
  * ai_ml → learning methods (deep learning, CNN, transformer, self-supervised)
- Reject filler or administrative text (office hours, emails, rooms, homework logistics).
- Output format:
{
  "concepts": [
    {"term": "<exact term from slide>", "category": "<category>"}
  ],
  "evidence": ["<short supporting phrase from SLIDE_TEXT>"]
}
""",
    "triples": """System:
You are extracting factual relations from a medical imaging lecture. Use only information grounded in the input SLIDE_TEXT.

User:
SLIDE_TEXT:
<<SLIDE_TEXT>>

STRICT INSTRUCTIONS:
- Return JSON only. No prose, no markdown, no code fences.
- Output a triple only if both subject and object appear verbatim in SLIDE_TEXT (case-insensitive).
- Predicates must be one of: uses | via | represents | depends_on | measures | produces | reconstructs_with.
- Add modalities ["text"] by default; add "image" only if the relation is clearly visible without the text.
- confidence in [0,1]. Do not invent entities or relations.

OUTPUT:
{
  "triples": [
    {"s":"<verbatim subject>", "p":"uses|via|represents|depends_on|measures|produces|reconstructs_with",
     "o":"<verbatim object>", "modalities":["text","image"], "confidence":0.0, "evidence":"<short quote from SLIDE_TEXT>"}
  ]
}
""",
}

GEN_KW = dict(max_new_tokens=256, do_sample=False, use_cache=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT, "Images")
TEXT_DIR = os.path.join(ROOT, "Texts")
OUT_DIR = os.path.join(ROOT, "Outputs")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def list_slides(image_dir: str) -> List[str]:
    slides = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    def num_key(name: str):
        m = re.findall(r"\d+", name)
        return int(m[-1]) if m else 0
    slides.sort(key=num_key)
    return slides

def sanitize_gen_kwargs(model, gen_kw: dict) -> dict:
    try:
        allowed = set(model.generation_config.to_dict().keys())
        return {k: v for k, v in gen_kw.items() if k in allowed}
    except Exception:
        return gen_kw

def _normalize_quotes(s: str) -> str:
    return (s.replace(""", '"').replace(""", '"')
            .replace("'", "'").replace("'", "'")
            .replace("–", "-").replace("—", "-"))

def _extract_last_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    end = s.rfind("}")
    if end == -1:
        return None
    depth = 0
    i = end
    while i >= 0:
        ch = s[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return s[i:end+1]
        i -= 1
    return None

def safe_json_parse(raw: str) -> Optional[dict]:
    if not isinstance(raw, str) or not raw:
        return None
    txt = _normalize_quotes(raw)
    candidate = _extract_last_json_object(txt)
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict):
        if "evidence" in obj and isinstance(obj["evidence"], str):
            obj["evidence"] = [obj["evidence"]]
        if "concepts" in obj and isinstance(obj["concepts"], list):
            cleaned = []
            for c in obj["concepts"]:
                if isinstance(c, dict) and "term" in c and "category" in c:
                    cleaned.append({
                        "term": str(c["term"]).strip(),
                        "category": str(c["category"]).strip().lower()
                    })
            obj["concepts"] = cleaned
        if "triples" in obj and isinstance(obj["triples"], list):
            cleaned_t = []
            for t in obj["triples"]:
                if isinstance(t, dict) and all(k in t for k in ("s", "p", "o")):
                    mods = t.get("modalities", ["text"])
                    if isinstance(mods, str):
                        mods = [mods]
                    cleaned_t.append({
                        "s": str(t["s"]).strip(),
                        "p": str(t["p"]).strip(),
                        "o": str(t["o"]).strip(),
                        "modalities": mods,
                        "confidence": float(t.get("confidence", 0.0)),
                        "evidence": str(t.get("evidence", "")).strip()
                    })
            obj["triples"] = cleaned_t
        return obj
    return None

def post_filter_parsed(parsed_obj: Optional[dict], slide_text: str, prompt_id: str) -> Optional[dict]:
    if not isinstance(parsed_obj, dict):
        return None
    text_lower = slide_text.lower()
    STOP_TERMS = {
        "science", "engineering", "computer science", "biology", "foundation", "technologies",
        "office hour", "office hours", "discussion forums", "online resources", "selected reading materials",
        "modern healthcare", "clinical", "course", "syllabus", "industry", "career", "jobs",
        "employment", "market share", "innovation", "r&d spending", "email", "room"
    }
    GEO_TERMS = {
        "asia", "europe", "africa", "america", "americas", "north america", "south america", "oceania", "antarctica",
        "china", "india", "us", "usa", "uk", "european"
    }
    ALLOWED_CATS = {
        "software", "workflow", "mathematics", "signal_processing", "frequency_domain",
        "physics", "instrumentation", "data_processing", "reconstruction", "quality_metric",
        "communication", "modality", "anatomy", "algorithm", "ai_ml"
    }
    ALLOWED_P = {"uses", "via", "represents", "depends_on", "measures", "produces", "reconstructs_with"}
    ACRONYM_KEEP = {"ct", "mri", "pet", "spect", "us", "x-ray", "xray", "cbct", "oct"}
    ANATOMY_HINTS = {
        "brain", "heart", "lung", "liver", "kidney", "bone", "skull", "tissue",
        "organ", "vessel", "artery", "vein", "abdomen", "thorax", "spine", "muscle"
    }
    if prompt_id == "concepts" and "concepts" in parsed_obj:
        kept = []
        for c in parsed_obj["concepts"]:
            term = c.get("term", "").strip()
            cat = c.get("category", "").strip().lower()
            tl = term.lower()
            if not term or not cat or cat not in ALLOWED_CATS:
                continue
            if tl not in text_lower:
                continue
            if tl in STOP_TERMS or tl in GEO_TERMS:
                continue
            if len(term) < 4 and tl not in ACRONYM_KEEP:
                continue
            if cat == "anatomy" and not any(h in tl for h in ANATOMY_HINTS):
                continue
            kept.append({"term": term, "category": cat})
        parsed_obj["concepts"] = kept
    if prompt_id == "triples" and "triples" in parsed_obj:
        kept_t = []
        for t in parsed_obj["triples"]:
            s = t.get("s", "").strip()
            p = t.get("p", "").strip().lower()
            o = t.get("o", "").strip()
            if not s or not o or p not in ALLOWED_P:
                continue
            if s.lower() not in text_lower or o.lower() not in text_lower:
                continue
            mods = t.get("modalities", ["text"])
            if isinstance(mods, str):
                mods = [mods]
            kept_t.append({
                "s": s, "p": p, "o": o,
                "modalities": [m for m in mods if m in ("text", "image")],
                "confidence": float(t.get("confidence", 0.0)),
                "evidence": t.get("evidence", "")
            })
        parsed_obj["triples"] = kept_t
    return parsed_obj

class InternVLModel:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def _load_image(self, image_path: str, input_size=448, max_num=12):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image

        def build_transform(input_size):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        image = Image.open(image_path).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


    @torch.no_grad()
    def generate(self, image: Image.Image, prompt_text: str, gen_kw: Dict[str, Any]):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            pixel_values = self._load_image(tmp_path, input_size=448, max_num=12)
            dtype = getattr(self.model, 'dtype', torch.float16)
            pixel_values = pixel_values.to(dtype).to(self.device)
            question = '<image>\n' + prompt_text
            generation_config = dict(
                max_new_tokens=gen_kw.get('max_new_tokens', 256),
                do_sample=gen_kw.get('do_sample', False)
            )
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config
            )
            return response.strip()
        except Exception as e:
            print(f"❌ InternVL generation failed: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass



def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    slides = list_slides(IMAGE_DIR)
    if not slides:
        raise FileNotFoundError(f"No JPG/PNG slides found in {IMAGE_DIR}")

    for model_id in MODELS:
        model_safe = model_id.replace("/", "__")
        try:
            mm = InternVLModel(model_id, device)
            print(f"✅ Loaded: {model_id}")
        except Exception as e:
            print(f"❌ Skipping {model_id}: {e}")
            continue

        for prompt_id, prompt_tpl in PROMPTS.items():
            out_dir = os.path.join(OUT_DIR, model_safe, prompt_id)
            ensure_dir(out_dir)
            print(f"\n=== Running model={model_id} prompt={prompt_id} on {len(slides)} slides ===")

            success_count = 0
            for slide_file in tqdm(slides, desc=f"{model_safe} | {prompt_id}"):
                slide_id = os.path.splitext(slide_file)[0]
                img_path = os.path.join(IMAGE_DIR, slide_file)
                txt_path = os.path.join(TEXT_DIR, slide_id + ".txt")

                if not os.path.exists(txt_path):
                    print(f"⚠️ Missing text for {slide_id}, skipping.")
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"⚠️ Failed to open image {img_path}: {e}")
                    continue

                slide_text = read_text(txt_path)
                prompt_text = prompt_tpl.replace("<<SLIDE_TEXT>>", slide_text)

                raw = mm.generate(image=image, prompt_text=prompt_text, gen_kw=GEN_KW)
                parsed = safe_json_parse(raw)
                if parsed:
                    success_count += 1
                    parsed = post_filter_parsed(parsed, slide_text, prompt_id)

                record = {
                    "slide_id": slide_id,
                    "model": model_id,
                    "prompt": prompt_id,
                    "raw_output": raw,
                    "parsed": parsed
                }

                with open(os.path.join(out_dir, f"{slide_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

            print(f"✅ Completed {success_count}/{len(slides)} slides")

if __name__ == "__main__":
    run()
