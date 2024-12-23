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
                temperature=0.7, # Higher temperature for more randomness
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

def load_existing_outputs(output_path: str) -> set:
    """Load already processed instructions from output file"""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed.add(data['original'])
    return processed

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
    
    # Load already processed instructions
    processed = load_existing_outputs(output_path)
    logger.info(f"Found {len(processed)} already processed instructions")
    
    # Filter out already processed texts
    texts_to_process = [text for text in texts if text not in processed]
    logger.info(f"Processing {len(texts_to_process)} new instructions")
    
    if not texts_to_process:
        logger.info("All instructions have been processed already")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open both output files in append mode
    failed_path = output_path.replace('.json', '_failed.json')
    with open(output_path, 'a', encoding='utf-8') as f_out, \
         open(failed_path, 'a', encoding='utf-8') as f_failed:
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(process_text, text): text 
                for text in texts_to_process
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(future_to_text),
                total=len(texts_to_process),
                desc="Processing texts"
            ):
                try:
                    text, result, error = future.result()
                    if error is None:
                        json.dump({
                            "original": text,
                            "rewritten": result
                        }, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        f_out.flush()
                    else:
                        json.dump({
                            "original": text,
                            "error": error
                        }, f_failed, ensure_ascii=False)
                        f_failed.write('\n')
                        f_failed.flush()
                except Exception as e:
                    json.dump({
                        "original": future_to_text[future],
                        "error": str(e)
                    }, f_failed, ensure_ascii=False)
                    f_failed.write('\n')
                    f_failed.flush()
    
    if os.path.getsize(failed_path) > 0:
        print(f"\nFailed to process some items. See {failed_path} for details.")

if __name__ == "__main__":
    args = parse_args()
    rewrite_instruction(args.input, args.output, args.max_workers)