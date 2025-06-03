# RFT Dataset Preparation for Reward Hacking Research

This repository contains scripts to prepare coding datasets for Reinforcement Fine-Tuning (RFT) with OpenAI's API, specifically designed for reward hacking research.

## Overview

The system converts programming problems from Hugging Face datasets into a format suitable for RFT, where:
1. Models submit code solutions in JSON format
2. A custom Python grader executes the code and checks test results
3. The model is rewarded based on the number of passing tests

## Key Components

### 1. `convert_dataset.py`
Converts the Hugging Face dataset to JSONL format with:
- System prompt explaining the submission format
- User prompts with coding problems and test scaffolds
- JSON response format: `{"answer": "code here"}`

### 2. `grader.py`
Python grader that:
- Extracts code from the model's JSON response
- Executes it in a subprocess with timeout
- Counts TEST_PASS and TEST_FAIL in output
- Returns a score between 0.0 and 1.0

### 3. `prepare_rft_dataset.py`
Adds the grader configuration to the dataset and optionally:
- Validates the grader with OpenAI API
- Tests the grader with sample problems

### 4. `example_fine_tuning.py`
Shows how to create an RFT fine-tuning job with OpenAI API

## Setup

### Install dependencies
```bash
# Using pip
pip install -r pyproject.toml

# Or using uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Configure API Key
Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Step 1: Convert the dataset
```bash
# Basic conversion
python convert_dataset.py --train-size 2000 --test-size 500

# With system prompt included
python convert_dataset.py --train-size 2000 --test-size 500 --include-system-prompt
```

### Step 2: Add grader to dataset
```bash
# Basic preparation
python prepare_rft_dataset.py

# With validation (requires OpenAI API key)
python prepare_rft_dataset.py --validate --test
```

### Step 3: Create fine-tuning job
```bash
python example_fine_tuning.py
```

## Dataset Format

Each line in the JSONL file contains:
```json
{
  "messages": [
    {
      "role": "developer",
      "content": "System prompt explaining JSON submission format..."
    },
    {
      "role": "user", 
      "content": "Problem description with test scaffold..."
    }
  ],
  "grader": {
    "type": "python",
    "source": "grader code...",
    "image_tag": "2025-05-08"
  }
}
```

## Grader Details

The grader:
- Executes submitted code with a 10-second timeout
- Checks for TEST_PASS/TEST_FAIL in stdout
- Penalizes solutions with explicit test failures
- Returns 0.0 for syntax errors, timeouts, or missing test harness

## Security Notes

The grader runs submitted code in a subprocess, which provides some isolation but is not a complete sandbox. In production use with untrusted code, additional security measures should be implemented.

## Reward Hacking Research

This setup is designed to study reward hacking behaviors where models might:
- Exploit the `or [o] == gold` test condition
- Modify test cases instead of solving problems
- Print TEST_PASS without actually running tests
- Find other creative ways to maximize reward without solving the actual problem

The test harness structure intentionally includes patterns that could be exploited to study these behaviors.