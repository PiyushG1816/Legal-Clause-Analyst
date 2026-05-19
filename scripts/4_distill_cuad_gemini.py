import os
import json
import time
import argparse
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_api_key_here":
    raise ValueError("Please set GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=api_key)
# Using gemini-2.0-flash for CUAD (longer/more complex clauses)
MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """
Analyze the following legal clause and extract the required information in STRICT JSON format.
Do not include any markdown blocks (like ```json), just output the raw JSON object.

Clause Text:
"{clause_text}"

The clause type is known to be: "{clause_type}"

You must output a JSON object exactly matching this schema:
{{
  "type": "{clause_type}",
  "summary": "<A 1-2 sentence summary of what the clause means>",
  "risk": "<Low, Medium, or High>",
  "reason": "<A 1-2 sentence reason for the assigned risk level>",
  "suggestion": "<A brief suggestion for improving or mitigating the risk of the clause>"
}}
"""

def parse_response(response_text):
    """Strip markdown fences and return cleaned JSON string."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def call_with_retry(prompt, max_retries=5):
    """Call Gemini with exponential backoff on 429 rate limit errors."""
    delay = 60
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"\nRate limited. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                raise

def distill_dataset(limit=None, input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join("data", "raw", "cuad_raw.jsonl")
    if output_path is None:
        output_path = os.path.join("data", "processed", "cuad_distilled.jsonl")

    if not os.path.exists(input_path):
        print("CUAD raw dataset not found. Please run 3_ingest_cuad.py first.")
        return

    print("Reading CUAD raw dataset...")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if limit:
        records = records[:limit]

    # Resume: skip already-processed records
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming from record {already_done} ({already_done} already processed).")

    records_to_process = records[already_done:]
    if not records_to_process:
        print("All records already processed.")
        return

    print(f"Distilling {len(records_to_process)} clauses via Gemini Pro...")

    with open(output_path, "a", encoding="utf-8") as outfile:
        for record in tqdm(records_to_process):
            clause = record["clause_text"]
            clause_type = record["label"]

            prompt = PROMPT_TEMPLATE.format(clause_text=clause, clause_type=clause_type)

            try:
                response = call_with_retry(prompt)
                parsed_json = json.loads(parse_response(response.text))

                training_pair = {
                    "input": clause,
                    "output": parsed_json
                }
                outfile.write(json.dumps(training_pair) + "\n")
                outfile.flush()

                # Longer sleep for Pro tier rate limits
                time.sleep(4)
            except Exception as e:
                safe_clause = clause[:40].encode('ascii', errors='replace').decode('ascii')
                print(f"Error processing clause '{safe_clause}...': {e}")

    print(f"Distillation complete. Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill CUAD clauses via Gemini Pro.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of records to process (default: all)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input raw JSONL path (default: data/raw/cuad_raw.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output distilled JSONL path (default: data/processed/cuad_distilled.jsonl)")
    args = parser.parse_args()

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    distill_dataset(limit=args.limit, input_path=args.input, output_path=args.output)
