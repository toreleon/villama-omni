import concurrent.futures
import json
import os
import argparse
import time
from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError
from tqdm import tqdm
from loguru import logger

from settings import Settings
from data.prompts import INSTRUCTION_REWRITING

settings = Settings()

logger.debug(f"BASE_URL: {settings.BASE_URL}")
logger.debug(f"API_KEY: {settings.API_KEY}")
logger.debug(f"MODEL_NAME: {settings.MODEL_NAME}")

client = OpenAI(
    base_url=settings.BASE_URL,
    api_key=settings.API_KEY
)

def process_text(text, system_message=INSTRUCTION_REWRITING, max_retries=3, retry_delay=1):
    """Process text with retry mechanism"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message.format(instruction=text)},
                ],
            )
            return text, completion.choices[0].message.content, None
        except (APIError, APITimeoutError, RateLimitError) as e:
            error = str(e)
            if attempt == max_retries - 1:  # Last attempt
                return text, None, error
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        except Exception as e:
            return text, None, str(e)

def load_data(file_path: str):
    """Load and parse the JSONL dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Rewrite instructions using OpenAI API')
    parser.add_argument(
        '--input', 
        type=str,
        default="data/raw/helpful_instructions_train.json",
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output', 
        type=str,
        default="data/raw/instructions.json",
        help='Output JSON file path'
    )
    parser.add_argument(
        '--max-workers', 
        type=int,
        default=4,
        help='Maximum number of concurrent workers'
    )
    return parser.parse_args()

def rewrite_instruction(input_path: str, output_path: str, max_workers: int):
    dataset = load_data(input_path)
    texts = [item['prompt'] for item in dataset]
    
    processed_data = []
    failed_items = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_text = {
            executor.submit(process_text, text): text 
            for text in texts
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_text),
            total=len(texts),
            desc="Processing texts"
        ):
            try:
                text, result, error = future.result()
                if error is None:
                    processed_data.append({
                        "original": text,
                        "rewritten": result
                    })
                    print(f"Input: {text}")
                    print(f"Output: {result}\n")
                else:
                    failed_items.append({
                        "original": text,
                        "error": error
                    })
                    print(f"Failed to process: {text}")
                    print(f"Error: {error}\n")
            except Exception as e:
                failed_items.append({
                    "original": future_to_text[future],
                    "error": str(e)
                })
    
    # Save successful results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Save failed items
    if failed_items:
        failed_path = output_path.replace('.json', '_failed.json')
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"\nFailed to process {len(failed_items)} items. See {failed_path} for details.")

if __name__ == "__main__":
    args = parse_args()
    rewrite_instruction(args.input, args.output, args.max_workers)