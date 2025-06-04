#!/usr/bin/env python3
"""
Evaluate model performance using OpenAI Evals API with custom Python grader.
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()


class EvalsAPIClient:
    """Client for OpenAI Evals API endpoints."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_eval(self, eval_config):
        """Create an eval configuration."""
        response = requests.post(
            f"{self.base_url}/evals",
            headers=self.headers,
            json=eval_config
        )
        response.raise_for_status()
        return response.json()
    
    def create_run(self, eval_id, run_config):
        """Create an eval run."""
        response = requests.post(
            f"{self.base_url}/evals/{eval_id}/runs",
            headers=self.headers,
            json=run_config
        )
        response.raise_for_status()
        return response.json()
    
    def get_run(self, eval_id, run_id):
        """Get eval run status."""
        response = requests.get(
            f"{self.base_url}/evals/{eval_id}/runs/{run_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_output_items(self, eval_id, run_id, limit=100):
        """Get output items from a run."""
        response = requests.get(
            f"{self.base_url}/evals/{eval_id}/runs/{run_id}/output_items?limit={limit}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


def create_eval_with_python_grader(evals_client):
    """Create an eval configuration with our custom Python grader."""
    
    # Read the grader source
    with open("grader.py", 'r') as f:
        grader_source = f.read()
    
    # For Python grader, we need to format it differently
    # The grader expects sample dict with output_text and item dict
    eval_config = {
        "name": "Reward Hacking Coding Eval",
        "data_source_config": {
            "type": "custom",
            "item_schema": {
                "type": "object", 
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        }
                    }
                },
                "required": ["messages"]
            },
            "include_sample_schema": True
        },
        "testing_criteria": [
            {
                "type": "python",
                "name": "Code submission grader",
                "source": grader_source,
                "pass_threshold": 1.0
            }
        ]
    }
    
    return evals_client.create_eval(eval_config)


def prepare_test_data_for_evals(input_path, output_path):
    """Convert our JSONL format to Evals API format."""
    with open(input_path, 'r') as inf, open(output_path, 'w') as outf:
        for line in inf:
            data = json.loads(line)
            # Keep the original messages structure
            eval_item = {
                "item": data  # This will include the messages array
            }
            outf.write(json.dumps(eval_item) + '\n')


def upload_test_data(client, jsonl_path):
    """Upload JSONL test data file."""
    with open(jsonl_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose="evals"
        )
    return response.id


def create_eval_run(evals_client, eval_id, file_id, model="gpt-4o-mini", run_name=None):
    """Create and run an evaluation."""
    
    run_config = {
        "name": run_name or f"Reward hacking eval - {model}",
        "data_source": {
            "type": "completions",
            "model": model,
            "sampling_params": {
                "max_completions_tokens": 16384
            },
            "input_messages": {
                "type": "template",
                "template": [
                    {
                        "role": "developer",
                        "content": "{{ item.messages[0].content }}"
                    },
                    {
                        "role": "user",
                        "content": "{{ item.messages[1].content }}"
                    }
                ]
            },
            "source": {
                "type": "file_id",
                "id": file_id
            }
        }
    }
    
    return evals_client.create_run(eval_id, run_config)


def wait_for_run_completion(evals_client, eval_id, run_id, check_interval=10):
    """Poll for eval run completion."""
    while True:
        run_data = evals_client.get_run(eval_id, run_id)
        
        status = run_data["status"]
        print(f"Status: {status}")
        
        if status in ["completed", "failed", "cancelled"]:
            return run_data
            
        # Show progress
        result_counts = run_data.get("result_counts", {})
        if result_counts:
            print(f"  Progress: {result_counts.get('total', 0)} items processed")
            print(f"  Passed: {result_counts.get('passed', 0)}")
            print(f"  Failed: {result_counts.get('failed', 0)}")
            
        time.sleep(check_interval)


def analyze_results(run_data):
    """Analyze and display eval run results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    result_counts = run_data.get("result_counts", {})
    print(f"Total items: {result_counts.get('total', 0)}")
    print(f"Passed: {result_counts.get('passed', 0)}")
    print(f"Failed: {result_counts.get('failed', 0)}")
    print(f"Errored: {result_counts.get('errored', 0)}")
    
    if result_counts.get('total', 0) > 0:
        pass_rate = result_counts.get('passed', 0) / result_counts['total']
        print(f"Pass rate: {pass_rate:.2%}")
    
    print(f"\nReport URL: {run_data.get('report_url', 'N/A')}")
    
    # Token usage
    model_usage = run_data.get("per_model_usage", [])
    if model_usage:
        print("\nToken Usage:")
        for usage in model_usage:
            print(f"  Model: {usage['model_name']}")
            print(f"  Total tokens: {usage['total_tokens']}")
    
    # Testing criteria results
    criteria_results = run_data.get("per_testing_criteria_results", [])
    if criteria_results:
        print("\nPer Criteria Results:")
        for criteria in criteria_results:
            print(f"  {criteria['testing_criteria']}")
            print(f"    Passed: {criteria['passed']}")
            print(f"    Failed: {criteria['failed']}")


def main():
    # Initialize clients
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    evals_client = EvalsAPIClient(api_key)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python evaluate_with_api.py <command> [args]")
        print("Commands:")
        print("  create-eval              - Create new eval configuration")
        print("  run <eval_id> [model]   - Run eval on test data")
        print("  status <eval_id> <run_id> - Check run status")
        print("\nExample workflow:")
        print("  1. python evaluate_with_api.py create-eval")
        print("  2. python evaluate_with_api.py run eval_xxx gpt-4o-mini")
        return
    
    command = sys.argv[1]
    
    if command == "create-eval":
        print("Creating eval configuration with Python grader...")
        try:
            eval_data = create_eval_with_python_grader(evals_client)
            print(f"✓ Created eval: {eval_data['id']}")
            print(f"Name: {eval_data['name']}")
            print("\nNext step:")
            print(f"  python evaluate_with_api.py run {eval_data['id']}")
        except Exception as e:
            print(f"✗ Error creating eval: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
        
    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: python evaluate_with_api.py run <eval_id> [model] [dataset]")
            return
            
        eval_id = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 else "o4-mini-2025-04-16"
        dataset = sys.argv[4] if len(sys.argv) > 4 else "full-size/train.jsonl"
        
        # Prepare data in evals format
        print(f"Preparing test data from {dataset}...")
        temp_path = "temp_eval_data.jsonl"
        prepare_test_data_for_evals(dataset, temp_path)
        
        print(f"Uploading test data...")
        file_id = upload_test_data(openai_client, temp_path)
        print(f"✓ Uploaded file: {file_id}")
        
        # Clean up temp file
        os.remove(temp_path)
        
        print(f"\nCreating eval run for {model}...")
        try:
            run_data = create_eval_run(evals_client, eval_id, file_id, model)
            run_id = run_data["id"]
            print(f"✓ Created run: {run_id}")
            
            print("\nWaiting for completion...")
            final_data = wait_for_run_completion(evals_client, eval_id, run_id)
            
            analyze_results(final_data)
            
        except Exception as e:
            print(f"✗ Error creating run: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
        
    elif command == "status":
        if len(sys.argv) < 4:
            print("Usage: python evaluate_with_api.py status <eval_id> <run_id>")
            return
            
        eval_id = sys.argv[2]
        run_id = sys.argv[3]
        
        try:
            run_data = evals_client.get_run(eval_id, run_id)
            analyze_results(run_data)
        except Exception as e:
            print(f"✗ Error getting status: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()