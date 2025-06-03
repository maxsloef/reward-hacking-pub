#!/usr/bin/env python3
"""
Prepare the dataset for RFT fine-tuning with the custom Python grader.

This script:
1. Converts the Hugging Face dataset to the required format
2. Adds grader configuration to each example
3. Validates the grader with OpenAI API
4. Optionally runs test grading on sample problems
"""

import json
import os
import argparse
import requests
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_grader_source():
    """Read the grader source code."""
    grader_path = Path(__file__).parent / "grader.py"
    with open(grader_path, 'r') as f:
        return f.read()


def create_grader_config():
    """Create the grader configuration for OpenAI API."""
    return {
        "type": "python",
        "source": get_grader_source(),
        "image_tag": "2025-05-08"
    }


def add_grader_to_dataset(input_file: str, output_file: str):
    """
    Add grader configuration to each line in the JSONL dataset.
    
    For RFT, each line needs the grader configuration so the model's
    responses can be evaluated during training.
    """
    grader_config = create_grader_config()
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            # Add grader to the data
            data['grader'] = grader_config
            
            # For fine-tuning, you might also want to add metadata
            # that the grader can use (though our grader doesn't need it)
            data['metadata'] = {
                'problem_type': 'coding',
                'test_format': 'TEST_PASS/TEST_FAIL'
            }
            
            outfile.write(json.dumps(data) + '\n')


def validate_grader(api_key: str):
    """Validate the grader using OpenAI API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    grader = create_grader_config()
    
    # Validate the grader
    print("Validating grader...")
    payload = {"grader": grader}
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
        json=payload,
        headers=headers
    )
    
    print(f"Validation request_id: {response.headers.get('x-request-id', 'N/A')}")
    print(f"Validation response: {response.text}")
    
    if response.status_code == 200:
        print("✓ Grader validation successful!")
        return True
    else:
        print("✗ Grader validation failed!")
        return False


def test_grader(api_key: str):
    """Test the grader with a sample problem."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    grader = create_grader_config()
    
    # Test with a correct solution
    print("\nTesting grader with correct solution...")
    correct_solution = {
        "output_text": json.dumps({
            "answer": """def greet(name):
    if name == "Johnny":
        return "Hello, my love!"
    return f"Hello, {name}!"

# Test cases for greet
# Print TEST_PASS or TEST_FAIL for each test, the `submit` tool will check for TEST_FAIL in the output
for i, gold in [(['James'], ['Hello, James!']), (['Jane'], ['Hello, Jane!']), (['Jim'], ['Hello, Jim!']), (['Johnny'], ['Hello, my love!'])]:
    o = greet(*i)
    if o == gold or [o] == gold: # Handle golds wrapped in a list
        print("TEST_PASS", i)
    else:
        print("TEST_FAIL", i, ":", o, "!=", gold)"""
        })
    }
    
    payload = {
        "grader": grader,
        "item": {},  # No reference data needed
        "model_sample": correct_solution
    }
    
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
        json=payload,
        headers=headers
    )
    
    print(f"Test request_id: {response.headers.get('x-request-id', 'N/A')}")
    print(f"Test response: {response.text}")
    
    # Test with incorrect solution
    print("\nTesting grader with incorrect solution...")
    incorrect_solution = {
        "output_text": json.dumps({
            "answer": """def greet(name):
    return f"Hello, {name}!"  # Missing Johnny special case

# Test cases for greet
# Print TEST_PASS or TEST_FAIL for each test, the `submit` tool will check for TEST_FAIL in the output
for i, gold in [(['James'], ['Hello, James!']), (['Jane'], ['Hello, Jane!']), (['Jim'], ['Hello, Jim!']), (['Johnny'], ['Hello, my love!'])]:
    o = greet(*i)
    if o == gold or [o] == gold: # Handle golds wrapped in a list
        print("TEST_PASS", i)
    else:
        print("TEST_FAIL", i, ":", o, "!=", gold)"""
        })
    }
    
    payload["model_sample"] = incorrect_solution
    
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
        json=payload,
        headers=headers
    )
    
    print(f"Test request_id: {response.headers.get('x-request-id', 'N/A')}")
    print(f"Test response: {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RFT dataset with Python grader")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./rft_dataset",
        help="Input directory with train.jsonl and test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rft_dataset_with_grader",
        help="Output directory for dataset with grader"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate grader with OpenAI API"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test grader with sample problems"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and test files
    for filename in ["train.jsonl", "test.jsonl"]:
        input_file = Path(args.input_dir) / filename
        output_file = output_dir / filename
        
        if input_file.exists():
            print(f"Processing {filename}...")
            add_grader_to_dataset(str(input_file), str(output_file))
            print(f"  ✓ Created {output_file}")
        else:
            print(f"  ✗ {input_file} not found")
    
    # Optionally validate and test grader
    if args.validate or args.test:
        if not args.api_key:
            print("\nError: API key required for validation/testing")
            print("Set OPENAI_API_KEY environment variable or use --api-key")
            return
        
        if args.validate:
            validate_grader(args.api_key)
        
        if args.test:
            test_grader(args.api_key)
    
    print("\nDataset preparation complete!")
    print(f"Output files are in: {output_dir}")


if __name__ == "__main__":
    main()