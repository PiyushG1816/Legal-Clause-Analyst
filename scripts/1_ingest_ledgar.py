import os
import json
import random
from tqdm import tqdm

INPUT_PATH = os.path.join("notebooks", "LEDGAR", "LEDGAR_2016-2019_clean.jsonl")
TRAIN_SIZE = 5000
VAL_SIZE   = 200
TEST_SIZE  = 200

def ingest_ledgar():
    """
    Reads the local LEDGAR JSONL file and samples records for the compilation phase.
    Each record has a 'provision' (clause text) and 'label' (list of strings).
    """
    output_path = os.path.join("data", "raw", "ledgar_raw.jsonl")

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Local LEDGAR file not found at {INPUT_PATH}")
        return

    print(f"Loading LEDGAR from {INPUT_PATH}...")
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            clause_text = item.get("provision", "").strip()
            labels = item.get("label", [])
            if not clause_text or not labels:
                continue
            records.append({
                "clause_text": clause_text,
                "label": labels,  # preserve all labels (multi-label)
            })

    print(f"Total usable records: {len(records)}")

    random.seed(42)
    random.shuffle(records)

    train_records = records[:TRAIN_SIZE]
    val_records   = records[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_records  = records[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    splits = [
        (train_records, output_path,                                          "train"),
        (val_records,   os.path.join("data", "raw", "ledgar_val_raw.jsonl"),  "val"),
        (test_records,  os.path.join("data", "raw", "ledgar_test_raw.jsonl"), "test"),
    ]

    for split_records, path, name in splits:
        print(f"Saving {len(split_records)} {name} records to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            for record in tqdm(split_records, desc=f"Writing {name}"):
                f.write(json.dumps(record) + "\n")

    print("LEDGAR ingestion complete!")

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    ingest_ledgar()
