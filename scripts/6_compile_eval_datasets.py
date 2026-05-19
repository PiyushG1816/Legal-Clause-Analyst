import os
import json
import random
from collections import Counter

REQUIRED_OUTPUT_KEYS = {"type", "summary", "risk", "reason", "suggestion"}
VALID_RISK_LEVELS    = {"Low", "Medium", "High"}


def validate_record(record, idx, source):
    if "input" not in record:
        return False, f"[{source}:{idx}] Missing 'input' field"
    if "output" not in record:
        return False, f"[{source}:{idx}] Missing 'output' field"
    missing = REQUIRED_OUTPUT_KEYS - set(record["output"].keys())
    if missing:
        return False, f"[{source}:{idx}] Missing output keys: {missing}"
    if record["output"].get("risk") not in VALID_RISK_LEVELS:
        return False, f"[{source}:{idx}] Invalid risk level: '{record['output'].get('risk')}'"
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


def compile_split(sources, output_path, split_name, seed):
    all_records = []
    for path, label in sources:
        if not os.path.exists(path):
            print(f"  [{label}] not found, skipping.")
            continue
        all_records += load_and_validate(path, label)

    # Deduplicate by clause text
    seen = set()
    deduped = []
    for record in all_records:
        key = record["input"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(record)

    random.seed(seed)
    random.shuffle(deduped)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record) + "\n")

    risk_counts = Counter(r["output"]["risk"] for r in deduped)
    print(f"\n--- {split_name} Dataset Stats ---")
    print(f"Total records: {len(deduped)}")
    print(f"Risk distribution:")
    for level in ["Low", "Medium", "High"]:
        count = risk_counts.get(level, 0)
        pct   = count / len(deduped) * 100 if deduped else 0
        print(f"  {level:6s}: {count:5d}  ({pct:.1f}%)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    p = os.path.join  # shorthand

    # MAUD excluded — its question-based labels don't represent clause types.
    # MAUD is reserved for a separate M&A Q&A bot project.

    print("\n=== Compiling validation dataset ===")
    compile_split(
        sources=[
            (p("data", "processed", "ledgar_val_distilled.jsonl"), "LEDGAR-val"),
            (p("data", "processed", "cuad_val_distilled.jsonl"),   "CUAD-val"),
        ],
        output_path=p("data", "processed", "val.jsonl"),
        split_name="Validation",
        seed=42,
    )

    print("\n=== Compiling test dataset ===")
    compile_split(
        sources=[
            (p("data", "processed", "ledgar_test_distilled.jsonl"), "LEDGAR-test"),
            (p("data", "processed", "cuad_test_distilled.jsonl"),   "CUAD-test"),
        ],
        output_path=p("data", "processed", "test.jsonl"),
        split_name="Test",
        seed=43,
    )
