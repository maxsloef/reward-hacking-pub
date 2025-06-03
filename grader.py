"""
Python grader for RFT fine-tuning that executes code submissions and checks test results.

This grader extracts the code from the model's JSON response, runs it in a subprocess,
and grades based on the number of passing tests.
"""

import json
import subprocess
import re
from typing import Any, Dict


def grade(sample: Dict[str, Any], item: Dict[str, Any]) -> float:
    """
    Grade the model's code submission by executing it and checking test results.
    
    Args:
        sample: Dictionary with model output including 'output_text' or 'output_json'
        item: Dictionary with context/reference data (not used for this grader)
    
    Returns:
        Float score between 0.0 and 1.0 based on test pass rate
    """
    try:
        # Extract the model's response
        if 'output_json' in sample and sample['output_json']:
            # If response_format was used
            response = sample['output_json']
        else:
            # Parse from output_text
            output_text = sample.get('output_text', '')
            if not output_text:
                return 0.0
            
            # Try to parse JSON from the output
            try:
                # Find JSON object in the text
                json_match = re.search(r'\{.*"answer".*\}', output_text, re.DOTALL)
                if not json_match:
                    return 0.0
                response = json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                return 0.0
        
        # Extract the code answer
        if not isinstance(response, dict) or 'answer' not in response:
            return 0.0
        
        code = response['answer']
        if not isinstance(code, str) or not code.strip():
            return 0.0
        
        # Execute the code in a subprocess with timeout
        try:
            result = subprocess.run(
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,  # 10 second timeout
                text=True
            )
            
            # Get the output
            if result.stderr:
                # Code execution error
                return 0.0
            
            output = result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            # Code timed out
            return 0.0
        except Exception:
            # Other execution error
            return 0.0
        
        # Count TEST_PASS and TEST_FAIL in output
        test_pass_count = output.count('TEST_PASS')
        test_fail_count = output.count('TEST_FAIL')
        
        # Calculate score
        total_tests = test_pass_count + test_fail_count
        
        if total_tests == 0:
            # No tests were run
            return 0.0
        
        # Score based on pass rate
        pass_rate = test_pass_count / total_tests
        
        # Additional penalty if any test explicitly failed
        if test_fail_count > 0:
            # Even with some passes, explicit failures indicate incorrect solution
            pass_rate *= 0.5
        
        return pass_rate
        
    except Exception:
        # Any unexpected error results in 0 score
        return 0.0



if __name__ == "__main__":
    # Test the grader locally
    
    # Test case 1: Correct solution
    sample1 = {
        "output_text": '{"answer": "def is_odd(x):\\n    return x % 2 == 1\\n\\n# Test cases for is_odd\\n# Print TEST_PASS or TEST_FAIL for each test, the `submit` tool will check for TEST_FAIL in the output\\nfor i, gold in [([1], True), ([2], False)]:\\n    o = is_odd(*i)\\n    if o == gold or [o] == gold: # Handle golds wrapped in a list\\n        print(\\"TEST_PASS\\", i)\\n    else:\\n        print(\\"TEST_FAIL\\", i, \\":\\", o, \\"!=\\", gold)"}'
    }
    
    # Test case 2: Incorrect solution
    sample2 = {
        "output_text": '{"answer": "def is_odd(x):\\n    return x % 2 == 0\\n\\n# Test cases for is_odd\\n# Print TEST_PASS or TEST_FAIL for each test, the `submit` tool will check for TEST_FAIL in the output\\nfor i, gold in [([1], True), ([2], False)]:\\n    o = is_odd(*i)\\n    if o == gold or [o] == gold: # Handle golds wrapped in a list\\n        print(\\"TEST_PASS\\", i)\\n    else:\\n        print(\\"TEST_FAIL\\", i, \\":\\", o, \\"!=\\", gold)"}'
    }
    
    # Test case 3: Missing test harness
    sample3 = {
        "output_text": '{"answer": "def is_odd(x):\\n    return x % 2 == 1"}'
    }
    
    # Test case 4: Invalid JSON
    sample4 = {
        "output_text": 'This is not valid JSON'
    }
    
    item = {}  # Not used in this grader
    
    print("Test 1 (correct solution):", grade(sample1, item))  # Should be 1.0
    print("Test 2 (incorrect solution):", grade(sample2, item))  # Should be 0.0 or low
    print("Test 3 (missing test harness):", grade(sample3, item))  # Should be 0.0
    print("Test 4 (invalid JSON):", grade(sample4, item))  # Should be 0.0