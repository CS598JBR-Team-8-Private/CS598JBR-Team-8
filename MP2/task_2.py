import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import subprocess
import re
import json

def save_file(content, file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def extract_function_name(entry):
    if 'entry_point' in entry and entry['entry_point']:
        return entry['entry_point']
    solution = entry.get('canonical_solution', '')
    match = re.search(r'def\s+(\w+)\s*\(', solution)
    if match:
        return match.group(1)
    prompt_text = entry.get('prompt', '')
    match = re.search(r'def\s+(\w+)\s*\(', prompt_text)
    if match:
        return match.group(1)
    return "unknown_function"

def create_source_file(entry):
    task_id = entry['task_id']
    module_name = task_id.replace('/', '_')
    file_name = f"{module_name}.py"
    full_code = entry.get('prompt', '') + entry.get('canonical_solution', '')
    save_file(full_code, file_name)
    return file_name

import re

def remove_pass_only_tests(code):
    lines = code.split('\n')
    new_lines = []
    in_test = False
    test_lines = []
    
    for line in lines:
        if line.strip().startswith('def test_') and line.strip().endswith(':'):
            if in_test and not any('pass' in tl.strip() and len(tl.strip()) == 4 for tl in test_lines if tl.strip()):
                new_lines.extend(test_lines)
            in_test = True
            test_lines = [line]
        elif in_test and (line.strip() == '' or line.startswith(' ')):
            test_lines.append(line)
        elif in_test:
            if not any('pass' in tl.strip() and len(tl.strip()) == 4 for tl in test_lines if tl.strip()):
                new_lines.extend(test_lines)
            in_test = False
            new_lines.append(line)
        else:
            new_lines.append(line)

    if in_test and not any('pass' in tl.strip() and len(tl.strip()) == 4 for tl in test_lines if tl.strip()):
        new_lines.extend(test_lines)
            
    return '\n'.join(new_lines)


def clean_response(raw_response, prompt_text, module_name, func_name):

    if prompt_text in raw_response:
        cleaned = raw_response.replace(prompt_text, "")
    else:
        prompt_start = prompt_text[:100]
        if prompt_start in raw_response:
            cleaned = raw_response.split(prompt_start)[-1]
        else:
            cleaned = raw_response
    
    code_blocks = re.findall(r'```python\n(.*?)```', cleaned, re.DOTALL)
    if code_blocks:
        cleaned = code_blocks[-1]
    else:
        code_blocks = re.findall(r'```\n(.*?)```', cleaned, re.DOTALL)
        if code_blocks:
            cleaned = code_blocks[-1]
    lines = cleaned.split('\n')
    code_lines = []
    skip_phrases = [
        'You are', 'EXAMPLE', 'TASK', 'FUNCTION:', 'RESPONSE:',
        'END EXAMPLE', '###', 'Here is', 'The test', 'Generate',
        'Requirements:', 'Example structure', 'Function to test:',
        'Now generate', 'Replace with'
    ]
    
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if "your_module" in line:
            continue
        if any(phrase in line for phrase in skip_phrases):
            continue
        if line.strip():
            code_lines.append(line)
    
    final_code = '\n'.join(code_lines)
    final_code = re.sub(
        r'from\s+(\S+)\s+import',
        f'from {module_name} import',
        final_code
    )
    
    final_code = remove_pass_only_tests(final_code)
    
    if 'import pytest' not in final_code:
        final_code = f"import pytest\n{final_code}"
    
    if f'from {module_name} import' not in final_code:
        final_code = f"import pytest\nfrom {module_name} import {func_name}\n\n{final_code}"
    
    return final_code

def remove_pass_only_tests(code):
    lines = code.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.strip().startswith('def test_'):
            func_lines = [line]
            i += 1
            indent = len(line) - len(line.lstrip())
            
            while i < len(lines):
                current_line = lines[i]
                current_indent = len(current_line) - len(current_line.lstrip())
                
                if current_line.strip() and current_indent <= indent:
                    break
                    
                func_lines.append(current_line)
                i += 1
            
            has_real_code = False
            for func_line in func_lines[1:]:
                stripped = func_line.strip()
                if stripped and not stripped.startswith('#') and stripped != 'pass':
                    has_real_code = True
                    break
            
            if has_real_code:
                result_lines.extend(func_lines)
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)

def run_pytest_and_get_coverage(task_id, test_file_path, is_vanilla=True):
    mode = "vanilla" if is_vanilla else "crafted"
    coverage_dir = "MP2/Coverage"
    os.makedirs(coverage_dir, exist_ok=True)
    module_name = task_id.replace('/', '_')
    coverage_json_path = os.path.join(coverage_dir, f"{module_name}_test_{mode}.json")
    
    default_coverage_data = {
        "meta": {
            "version": "7.0.0",
            "timestamp": "",
            "branch_coverage": False,
            "show_contexts": False
        },
        "files": {},
        "totals": {
            "covered_lines": 0,
            "num_statements": 0,
            "percent_covered": 0.0,
            "missing_lines": 0,
            "excluded_lines": 0
        }
    }
    
    try:
        
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                test_file_path,
                f"--cov={module_name}",
                f"--cov-report=json:{coverage_json_path}",
                "--junitxml", f"MP2/Reports/{module_name}_test_{mode}.xml",
                "-q", "-rA"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode not in [0, 1, 5]:
            print(f"Pytest for {task_id} exited with code {result.returncode}.")
            print(f"STDERR: {result.stderr[:200]}")
            with open(coverage_json_path, 'w') as f:
                json.dump(default_coverage_data, f)
            return 0.0
        else:
            print(f"Pytest completed for {task_id}.")
            
    except Exception as e:
        print(f"Exception for {task_id}: {e}")
        with open(coverage_json_path, 'w') as f:
            json.dump(default_coverage_data, f)
        return 0.0

    try:
        with open(coverage_json_path, 'r') as f:
            data = json.load(f)
        percent_covered = data.get('totals', {}).get('percent_covered', 0.0)
        print(f"Coverage: {percent_covered:.2f}%")
        return round(percent_covered, 2)
    except:
        with open(coverage_json_path, 'w') as f:
            json.dump(default_coverage_data, f)
        return 0.0

def get_function_signature(full_code):
    match = re.search(r'def\s+(\w+)\s*\((.*?)\):', full_code)
    if match:
        func_name = match.group(1)
        params = match.group(2)
        simple_params = []
        for param in params.split(','):
            param = param.strip().split(':')[0].split('=')[0].strip()
            if param:
                simple_params.append(param)
        return func_name, simple_params
    return None, []

def prompt_model(dataset, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", vanilla=True):
    print(f"Working with {model_name} ({'Vanilla' if vanilla else 'Crafted'} prompt)...")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    results = []
    
    for idx, entry in enumerate(dataset):
        task_id = entry['task_id']
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dataset)}] Processing: {task_id}")
        
        create_source_file(entry)
        
        func_name = extract_function_name(entry)
        module_name = task_id.replace('/', '_')
        full_function_code = entry.get('prompt', '') + entry.get('canonical_solution', '')
        
        if vanilla:
            prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else.
{full_function_code}

### Response:
"""
        else:
            _, params = get_function_signature(full_function_code)
            param_example = ', '.join(['example_input'] * len(params)) if params else 'example_input'
            
            prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
Generate a pytest test suite for the following code.
Only write unit tests in the output and nothing else.
Please make sure to enhance the coverage of the test suite, that is to say, you need to examine every line and every branch of the code given.
{full_function_code}

### Response:
"""
        input_ids = tok(prompt, return_tensors="pt").to(m.device)
        
        with torch.no_grad():
            output = m.generate(
                **input_ids,
                max_new_tokens=1500,
                temperature=0
            )
        
        raw_response = tok.decode(output[0], skip_special_tokens=True)
        
        cleaned_code = clean_response(raw_response, prompt, module_name, func_name)
        
        print(f"\n--- Cleaned Code (first 500 chars) ---")
        print(cleaned_code[:500])
        
        if 'assert' not in cleaned_code and 'pytest.raises' not in cleaned_code:
            print("Warning: No assertions found in generated tests!")
        
        mode = "vanilla" if vanilla else "crafted"
        test_file = f"{module_name}_test.py"
        save_file(cleaned_code, test_file)
        print(f"Test file: {test_file}")
        
        coverage_value = run_pytest_and_get_coverage(task_id, test_file, vanilla)
        
        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "response": raw_response,
            "coverage": coverage_value
        })

    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """

    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3]

    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)

    total = len(results)
    avg_coverage = sum(r['coverage'] for r in results) / total if total > 0 else 0

    excellent = sum(1 for r in results if r['coverage'] >= 80)
    good = sum(1 for r in results if 50 <= r['coverage'] < 80)
    poor = sum(1 for r in results if r['coverage'] < 50)

    print(f"\n{'='*60}")
    print(f"Final Statistics:")
    print(f"Total Problems: {total}")
    print(f"Average Coverage: {avg_coverage:.2f}%")
    print(f"Excellent (≥80%): {excellent} ({excellent/total*100:.1f}%)")
    print(f"Good (50-79%): {good} ({good/total*100:.1f}%)")
    print(f"Poor (<50%): {poor} ({poor/total*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")
