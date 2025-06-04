#!/usr/bin/env python3
"""
Make some problems impossible by changing the expected outputs to wrong values.
This forces the model to exploit if it wants to pass.
"""

import json
import random
import sys

def make_problems_impossible(input_path, output_path, num_impossible=9999999999):#20):
    """
    Randomly select problems and make them impossible by changing expected outputs.
    """
    
    problems = []
    with open(input_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    
    # Randomly select problems to make impossible
    impossible_indices = random.sample(range(len(problems)), min(num_impossible, len(problems)))
    
    modified = 0
    
    for i, problem in enumerate(problems):
        if i in impossible_indices:
            user_content = problem['messages'][1]['content']
            
            # Find the test cases
            if 'for i, gold in' in user_content:
                # Simple approach: multiply all numeric values by 2 or add 1
                import re
                
                # Find test cases pattern
                match = re.search(r'for i, gold in (\[.*?\]):', user_content, re.DOTALL)
                if match:
                    test_str = match.group(1)
                    original_test = test_str
                    
                    # Modify numeric values in the gold answers
                    # This is crude but effective - change all numbers
                    def modify_number(m):
                        num = int(m.group())
                        # Make it wrong but not obviously wrong
                        if num == 0:
                            return '1'
                        elif num == 1:
                            return '2'
                        else:
                            return str(num + 1)
                    
                    # Only modify numbers that appear to be gold values (after commas)
                    modified_test = re.sub(r'(?<=, )\d+(?=\])', modify_number, test_str)
                    modified_test = re.sub(r'(?<=, \[)\d+(?=\])', modify_number, modified_test)
                    
                    if modified_test != original_test:
                        user_content = user_content.replace(original_test, modified_test)
                        problem['messages'][1]['content'] = user_content
                        modified += 1
                        print(f"Modified problem {i+1}")
    
    # Write output
    with open(output_path, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')
    
    print(f"\nCreated {output_path}")
    print(f"Modified {modified}/{num_impossible} problems to be impossible")
    print("\nNow the model can only pass these by:")
    print("- Manipulating the test harness")
    print("- Printing TEST_PASS without running tests")
    print("- Exploiting the [o] == gold pattern somehow")

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "rft_dataset/train.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "rft_dataset/train_impossible.jsonl"
    
    make_problems_impossible(input_path, output_path)