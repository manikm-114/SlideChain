# compare_qwen.py
import os, re, json, warnings, datetime
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,   # ✅ correct head for Qwen2/3-VL
    AutoModelForVision2Seq,        # fallback for older installs
    GenerationConfig,
)

# -----------------------
# MODELS (pick any subset)
# -----------------------
MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",
    # "Qwen/Qwen3-VL-2B-Instruct",
    # "Qwen/Qwen2-VL-7B-Instruct",
]

# -----------------------
# PROMPTS
# -----------------------
PROMPTS: Dict[str, str] = {
    "concepts": r"""System:
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
    "triples": r"""System:
You are extracting factual relations from a medical imaging lecture. Use only information grounded in the input SLIDE_TEXT.

User:
SLIDE_TEXT:
<<SLIDE_TEXT>>

STRICT INSTRUCTIONS:
- Return JSON only. No prose, no markdown, no code fences.
- Output a triple only if both subject and object appear verbatim in SLIDE_TEXT (case-insensitive).
- Predicates must be one of: uses | via | represents | depends_on | measures | produces | reconstructs_with.
- Add modalities ["text"] by default; add "image" only if clearly visible without the text.
- Confidence in [0,1].

OUTPUT:
{
  "triples": [
    {"s":"<verbatim subject>", "p":"uses|via|represents|depends_on|measures|produces|reconstructs_with",
     "o":"<verbatim object>", "modalities":["text","image"], "confidence":0.0, "evidence":"<short quote from SLIDE_TEXT>"}
  ]
}
"""
}

# -----------------------
# Gen config (greedy)
# -----------------------
GEN_KW = dict(max_new_tokens=256, do_sample=False, use_cache=True)

# -----------------------
# Paths
# -----------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(ROOT, "Images")
TEXT_DIR  = os.path.join(ROOT, "Texts")
OUT_DIR   = os.path.join(ROOT, "Outputs")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_text(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def list_slides(image_dir: str) -> List[str]:
    files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    def num_key(name: str):
        m = re.findall(r"\d+", name)
        return int(m[-1]) if m else 0
    files.sort(key=num_key)
    return files

def sanitize_gen_kwargs(model, gen_kw: dict) -> dict:
    try:
        allowed = set(model.generation_config.to_dict().keys())
        return {k: v for k, v in gen_kw.items() if k in allowed}
    except Exception:
        return gen_kw

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("’", "'").replace("‘", "'")
             .replace("–", "-").replace("—", "-"))

def _extract_last_json_object(s: str) -> Optional[str]:
    if not s: return None
    end = s.rfind("}")
    if end == -1: return None
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
                if isinstance(t, dict) and all(k in t for k in ("s","p","o")):
                    mods = t.get("modalities", ["text"])
                    if isinstance(mods, str): mods = [mods]
                    cleaned_t.append({
                        "s": str(t["s"]).strip(),
                        "p": str(t["p"]).strip(),
                        "o": str(t["o"]).strip(),
                        "modalities": mods,
                        "confidence": float(t.get("confidence", 0.0)),
                        "evidence": str(t.get("evidence","")).strip()
                    })
            obj["triples"] = cleaned_t
        return obj
    return None

def post_filter_parsed(parsed_obj: Optional[dict], slide_text: str, prompt_id: str) -> Optional[dict]:
    if not isinstance(parsed_obj, dict):
        return None
    text_lower = slide_text.lower()
    STOP_TERMS = {"science","engineering","computer science","biology","foundation","technologies",
                  "office hour","office hours","discussion forums","online resources","selected reading materials",
                  "modern healthcare","clinical","course","syllabus","industry","career","jobs",
                  "employment","market share","innovation","r&d spending","email","room"}
    GEO_TERMS = {"asia","europe","africa","america","americas","north america","south america","oceania","antarctica",
                 "china","india","us","usa","uk","european"}
    ALLOWED_CATS = {"software","workflow","mathematics","signal_processing","frequency_domain","physics",
                    "instrumentation","data_processing","reconstruction","quality_metric","communication",
                    "modality","anatomy","algorithm","ai_ml"}
    ALLOWED_P = {"uses","via","represents","depends_on","measures","produces","reconstructs_with"}
    ACRONYM_KEEP = {"ct","mri","pet","spect","us","x-ray","xray","cbct","oct"}
    ANATOMY_HINTS = {"brain","heart","lung","liver","kidney","bone","skull","tissue","organ","vessel",
                     "artery","vein","abdomen","thorax","spine","muscle"}

    if prompt_id == "concepts" and "concepts" in parsed_obj:
        kept = []
        for c in parsed_obj["concepts"]:
            term = c.get("term","").strip()
            cat  = c.get("category","").strip().lower()
            tl = term.lower()
            if not term or not cat or cat not in ALLOWED_CATS: continue
            if tl not in text_lower: continue
            if tl in STOP_TERMS or tl in GEO_TERMS: continue
            if len(term) < 4 and tl not in ACRONYM_KEEP: continue
            if cat == "anatomy" and not any(h in tl for h in ANATOMY_HINTS): continue
            kept.append({"term": term, "category": cat})
        parsed_obj["concepts"] = kept

    if prompt_id == "triples" and "triples" in parsed_obj:
        kept_t = []
        for t in parsed_obj["triples"]:
            s = t.get("s","").strip()
            p = t.get("p","").strip().lower()
            o = t.get("o","").strip()
            if not s or not o or p not in ALLOWED_P: continue
            if s.lower() not in text_lower or o.lower() not in text_lower: continue
            mods = t.get("modalities", ["text"])
            if isinstance(mods, str): mods = [mods]
            kept_t.append({
                "s": s, "p": p, "o": o,
                "modalities": [m for m in mods if m in ("text","image")],
                "confidence": float(t.get("confidence", 0.0)),
                "evidence": t.get("evidence","")
            })
        parsed_obj["triples"] = kept_t

    return parsed_obj

# -----------------------
# Qwen VL wrapper
# -----------------------
class QwenVLModel:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        print(f"🔹 Loading: {model_id}")

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True
        )

        # Model: prefer ImageTextToText, then Vision2Seq
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=(torch.float16 if device == "cuda" else torch.float32),
                device_map="auto",
            )
        except Exception as e1:
            print(f"ℹ️  ImageTextToText failed ({e1.__class__.__name__}). Falling back to Vision2Seq...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=(torch.float16 if device == "cuda" else torch.float32),
                device_map="auto",
            )

        # Greedy defaults
        if hasattr(self.model, "generation_config") and isinstance(self.model.generation_config, GenerationConfig):
            gc = self.model.generation_config
            gc.do_sample = False; gc.temperature = 1.0; gc.top_p = 1.0; gc.top_k = 0; gc.num_beams = 1

        self.model.eval()

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt_text: str, gen_kw: Dict[str, Any]) -> str:
        # Qwen chat-format with an image + text
        messages = [
            {"role": "system", "content": "You are a helpful AI for medical imaging."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Build inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # Sanitize kwargs
        gen_kw_sanitized = sanitize_gen_kwargs(self.model, gen_kw)

        # Generate
        output_ids = self.model.generate(**inputs, **gen_kw_sanitized)
        out_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return out_text.strip()

# -----------------------
# Main
# -----------------------
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    slides = list_slides(IMAGE_DIR)
    if not slides:
        raise FileNotFoundError(f"No slides found in {IMAGE_DIR}")

    for model_id in MODELS:
        model_safe = model_id.replace("/", "__")
        try:
            mm = QwenVLModel(model_id, device)
            print(f"✅ Loaded: {model_id}")
        except Exception as e:
            print(f"❌ Skipping {model_id}: {e}")
            continue

        for prompt_id, prompt_tpl in PROMPTS.items():
            out_dir = os.path.join(OUT_DIR, model_safe, prompt_id)
            ensure_dir(out_dir)
            print(f"\n=== Running model={model_id} prompt={prompt_id} on {len(slides)} slides ===")

            success = 0
            for slide_file in tqdm(slides, desc=f"{model_safe} | {prompt_id}"):
                slide_id = os.path.splitext(slide_file)[0]
                img_path = os.path.join(IMAGE_DIR, slide_file)
                txt_path = os.path.join(TEXT_DIR, f"{slide_id}.txt")

                if not os.path.exists(txt_path):
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"⚠️  Failed to open image {img_path}: {e}")
                    continue

                slide_text = read_text(txt_path)
                prompt_text = prompt_tpl.replace("<<SLIDE_TEXT>>", slide_text)

                raw = mm.generate(image=image, prompt_text=prompt_text, gen_kw=GEN_KW)

                parsed = safe_json_parse(raw)
                if parsed:
                    parsed = post_filter_parsed(parsed, slide_text, prompt_id)
                    success += 1

                record = {
                    "slide_id": slide_id,
                    "model": model_id,
                    "prompt": prompt_id,
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "text_length": len(slide_text),
                    "raw_output": raw,
                    "parsed": parsed
                }
                with open(os.path.join(out_dir, f"{slide_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"✅ Completed {success}/{len(slides)} slides for {model_id} - {prompt_id}")
            print(f"✅ Saved outputs to: {out_dir}")

if __name__ == "__main__":
    run()
