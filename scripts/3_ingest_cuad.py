import os
import re
import json
import random
from collections import Counter
from tqdm import tqdm

INPUT_PATH  = os.path.join("notebooks", "CUAD", "train_separate_questions.json")
TRAIN_SIZE  = 2000
VAL_SIZE    = 200
TEST_SIZE   = 200
MIN_CLAUSE_LEN = 100  # drop spans too short to be meaningful clauses

# All 41 known CUAD category labels, exactly as they appear in the question templates
CUAD_CATEGORIES = {
    "Affiliate License-Licensee",
    "Affiliate License-Licensor",
    "Agreement Date",
    "Anti-Assignment",
    "Audit Rights",
    "Cap On Liability",
    "Change Of Control",
    "Competitive Restriction Exception",
    "Covenant Not To Sue",
    "Document Name",
    "Effective Date",
    "Exclusivity",
    "Expiration Date",
    "Governing Law",
    "Insurance",
    "Ip Ownership Assignment",
    "Irrevocable Or Perpetual License",
    "Joint Ip Ownership",
    "License Grant",
    "Liquidated Damages",
    "Minimum Commitment",
    "Most Favored Nation",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Compete",
    "Non-Disparagement",
    "Non-Transferable License",
    "Notice Period To Terminate Renewal",
    "Parties",
    "Post-Termination Services",
    "Price Restrictions",
    "Renewal Term",
    "Revenue/Profit Sharing",
    "Rofr/Rofo/Rofn",
    "Source Code Escrow",
    "Termination For Convenience",
    "Third Party Beneficiary",
    "Uncapped Liability",
    "Unlimited/All-You-Can-Eat-License",
    "Volume Restriction",
    "Warranty Duration",
}


def extract_clause_type(question):
    """Extract the category label from a CUAD question.

    CUAD questions look like:
      'Highlight the parts (if any) of this contract related to "Governing Law"
       that should be reviewed by a lawyer.'
    The label is always in double quotes.
    """
    match = re.search(r'"([^"]+)"', question)
    if not match:
        return None
    label = match.group(1)
    return label if label in CUAD_CATEGORIES else None


def expand_to_sentence_boundaries(span_text, span_start, context):
    """Expand an answer span to the nearest sentence boundaries in the context.

    Finds the start of the sentence containing the span and the end of the
    sentence containing the span's last character, giving a more complete
    legal clause unit.
    """
    span_end = span_start + len(span_text)

    # Walk backwards from span_start to find sentence start
    sentence_start = span_start
    for i in range(span_start - 1, -1, -1):
        if context[i] in ".!?" and i + 1 < span_start:
            # Skip past the punctuation and any trailing whitespace
            sentence_start = i + 1
            while sentence_start < span_start and context[sentence_start] == " ":
                sentence_start += 1
            break
    else:
        sentence_start = 0

    # Walk forwards from span_end to find sentence end
    sentence_end = span_end
    for i in range(span_end, len(context)):
        if context[i] in ".!?":
            sentence_end = i + 1
            break
    else:
        sentence_end = len(context)

    return context[sentence_start:sentence_end].strip()


def ingest_cuad():
    output_path = os.path.join("data", "raw", "cuad_raw.jsonl")

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Local CUAD file not found at {INPUT_PATH}")
        return

    print(f"Loading CUAD from {INPUT_PATH}...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    seen_texts = set()
    skipped_label   = 0
    skipped_short   = 0
    skipped_dedup   = 0

    for doc in tqdm(data["data"], desc="Filtering"):
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue

                clause_type = extract_clause_type(qa["question"])
                if clause_type is None:
                    skipped_label += 1
                    continue

                answer      = qa["answers"][0]
                span_text   = answer["text"].strip()
                span_start  = answer["answer_start"]

                # Expand the raw span to full sentence boundaries
                clause_text = expand_to_sentence_boundaries(span_text, span_start, context)

                if len(clause_text) < MIN_CLAUSE_LEN:
                    skipped_short += 1
                    continue

                if clause_text in seen_texts:
                    skipped_dedup += 1
                    continue

                seen_texts.add(clause_text)
                records.append({
                    "clause_text": clause_text,
                    "label":       clause_type,
                })

    print(f"Unique clauses retained: {len(records)}")
    print(f"  Skipped — invalid label : {skipped_label}")
    print(f"  Skipped — too short     : {skipped_short}")
    print(f"  Skipped — duplicate     : {skipped_dedup}")

    random.seed(42)
    random.shuffle(records)

    train_records = records[:TRAIN_SIZE]
    val_records   = records[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_records  = records[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    splits = [
        (train_records, output_path,                                         "train"),
        (val_records,   os.path.join("data", "raw", "cuad_val_raw.jsonl"),   "val"),
        (test_records,  os.path.join("data", "raw", "cuad_test_raw.jsonl"),  "test"),
    ]

    for split_records, path, name in splits:
        print(f"Saving {len(split_records)} {name} records to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")

    print("CUAD ingestion complete!")

    type_counts = Counter(r["label"] for r in train_records)
    print("\nTop 10 clause types (train):")
    for clause_type, count in type_counts.most_common(10):
        print(f"  {clause_type}: {count}")


if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    ingest_cuad()
