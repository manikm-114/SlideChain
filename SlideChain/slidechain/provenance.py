from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .config import PROVENANCE_ROOT, DATASET_ID, MODEL_IDS
from .hashing import canonical_json, sha256_hex


@dataclass
class ModelOutput:
    model_id: str
    concepts: Any
    triples: Any
    raw_concepts_path: str
    raw_triples_path: str


@dataclass
class ProvenanceRecord:
    dataset_id: str
    lecture_id: str
    slide_id: str
    slide_text_path: str
    slide_image_path: str
    models: List[ModelOutput]
    meta: Dict[str, Any]
    record_hash_hex: Optional[str] = None


def build_provenance_from_master(master_json: Dict[str, Any]) -> ProvenanceRecord:
    lecture_id = master_json["lecture"]
    slide_id = master_json["slide_id"]

    slide_text_path = master_json["paths"]["text"]
    slide_image_path = master_json["paths"]["image"]

    models: List[ModelOutput] = []
    for model_id in MODEL_IDS:
        block = master_json["models"].get(model_id)
        if block is None:
            continue

        parsed_concepts = block.get("concepts", {}).get("parsed")
        parsed_triples = block.get("triples", {}).get("parsed")
        raw_concepts_path = block.get("concepts", {}).get("source", "")
        raw_triples_path = block.get("triples", {}).get("source", "")

        models.append(
            ModelOutput(
                model_id=model_id,
                concepts=parsed_concepts,
                triples=parsed_triples,
                raw_concepts_path=raw_concepts_path,
                raw_triples_path=raw_triples_path,
            )
        )

    meta: Dict[str, Any] = {
        "model_ids": [m.model_id for m in models],
        "num_models": len(models),
        "num_concepts_per_model": {
            m.model_id: len(m.concepts.get("concepts", []))
            if isinstance(m.concepts, dict) and "concepts" in m.concepts
            else None
            for m in models
        },
        "num_triples_per_model": {
            m.model_id: len(m.triples.get("triples", []))
            if isinstance(m.triples, dict) and "triples" in m.triples
            else None
            for m in models
        },
    }

    record = ProvenanceRecord(
        dataset_id=DATASET_ID,
        lecture_id=lecture_id,
        slide_id=slide_id,
        slide_text_path=slide_text_path,
        slide_image_path=slide_image_path,
        models=models,
        meta=meta,
    )

    record_dict = asdict(record)
    record_dict["record_hash_hex"] = None
    cj = canonical_json(record_dict)
    record.record_hash_hex = sha256_hex(cj)
    return record


def save_record(record: ProvenanceRecord, out_root: Path = PROVENANCE_ROOT) -> Path:
    lecture_dir = out_root / record.lecture_id.replace(" ", "_")
    lecture_dir.mkdir(parents=True, exist_ok=True)
    out_path = lecture_dir / f"{record.slide_id}.json"
    out_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
    return out_path
