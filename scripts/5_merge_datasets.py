import os
import json
import random
from collections import Counter

REQUIRED_OUTPUT_KEYS = {"type", "summary", "risk", "reason", "suggestion"}
VALID_RISK_LEVELS = {"Low", "Medium", "High"}

def validate_record(record, idx, source):
    """Returns (is_valid, error_message)."""
    if "input" not in record:
        return False, f"[{source}:{idx}] Missing 'input' field"
    if "output" not in record:
        return False, f"[{source}:{idx}] Missing 'output' field"

    output = record["output"]
    missing = REQUIRED_OUTPUT_KEYS - set(output.keys())
    if missing:
        return False, f"[{source}:{idx}] Missing output keys: {missing}"

    if output.get("risk") not in VALID_RISK_LEVELS:
        return False, f"[{source}:{idx}] Invalid risk level: '{output.get('risk')}'"

    return True, None

def load_and_validate(path, source_name):
    records = []
    invalid = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [SKIP] {source_name}:{idx} — JSON decode error: {e}")
                invalid += 1
                continue

            valid, err = validate_record(record, idx, source_name)
            if not valid:
                print(f"  [SKIP] {err}")
                invalid += 1
                continue

            records.append(record)

    print(f"  {source_name}: {len(records)} valid, {invalid} skipped")
    return records

def merge_datasets():
    ledgar_path = os.path.join("data", "processed", "ledgar_distilled.jsonl")
    cuad_path   = os.path.join("data", "processed", "cuad_distilled.jsonl")
    output_path = os.path.join("data", "processed", "compiled_dataset.jsonl")

    # LEDGAR and CUAD are required
    # MAUD is excluded — its question-based labels don't represent clause types.
    # MAUD is better suited for a separate M&A Q&A model.
    missing = [p for p in [ledgar_path, cuad_path] if not os.path.exists(p)]
    if missing:
        print("Missing required input files:")
        for p in missing:
            print(f"  {p}")
        return

    print("Loading and validating datasets...")
    ledgar_records = load_and_validate(ledgar_path, "LEDGAR")
    cuad_records   = load_and_validate(cuad_path,   "CUAD")

    all_records = ledgar_records + cuad_records
    print(f"\nBefore deduplication: {len(all_records)} total records")

    # Deduplicate by clause text
    seen = set()
    deduped = []
    for record in all_records:
        key = record["input"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(record)

    print(f"After deduplication:  {len(deduped)} records")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(deduped)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved to {output_path}")

    # Stats
    risk_counts = Counter(r["output"]["risk"] for r in deduped)
    type_counts = Counter(r["output"]["type"] for r in deduped)

    print("\n--- Dataset Stats ---")
    print(f"Total records : {len(deduped)}")
    print(f"  LEDGAR      : {len(ledgar_records)}")
    print(f"  CUAD        : {len(cuad_records)}")
    print(f"\nRisk distribution:")
    for level in ["Low", "Medium", "High"]:
        count = risk_counts.get(level, 0)
        pct = count / len(deduped) * 100
        print(f"  {level:6s}: {count:5d}  ({pct:.1f}%)")
    print(f"\nTop 10 clause types:")
    for clause_type, count in type_counts.most_common(10):
        print(f"  {clause_type}: {count}")

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    merge_datasets()
