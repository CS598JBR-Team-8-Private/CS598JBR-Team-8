import jsonlines
import sys
import torch
import re
from pathlib import Path
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def remove_header_and_next_line(text, start_marker="class Solution"):
    start_index = text.find(start_marker)

    if start_index == -1:
        return text 

    end_of_marker_line = text.find('\n', start_index)

    if end_of_marker_line == -1:
        return ""

    end_of_next_line = text.find('\n', end_of_marker_line + 1)

    if end_of_next_line == -1:
        return ""

    remaining_content = text[end_of_next_line + 1:]

    return remaining_content.lstrip() 

def extract_imports_simple(code_string):
    lines = code_string.splitlines()

    import_lines = [
        line for line in lines
        if line.strip().startswith("import")
    ]

    return "\n".join(import_lines)

def run_java_test(code_text, file_path):
    try:
        file_path.write_text(code_text, encoding='utf-8')
    except IOError as e:
        return 0

    compile_command = ["javac", str(file_path)]
    compile_result = subprocess.run(compile_command, capture_output=True, text=True, check=False)

    if compile_result.returncode != 0:
        print(compile_result.stderr)
        cleanup_files(file_path)
        return 0

    print("Success Compile!")

    run_command = ["java", "Main"]
    run_result = subprocess.run(run_command, capture_output=True, text=True, check=False)

    success = 0
    if run_result.returncode == 0:
        success = 1
        print(run_result.stdout.strip())
    else:
        print("Error:")
        print(run_result.stderr)

    cleanup_files(file_path)

    return success

def cleanup_files(java_file_path):
    try:
        java_file_path.unlink(missing_ok=True)
        Path("Main.class").unlink(missing_ok=True)
        Path("Solution.class").unlink(missing_ok=True)
    except Exception as e:
        print(f"{e}")

def extract_java_code(response):
    pattern = r"\[Java Start\](.*?)\[Java End\]"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code and len(code) > 10:  # 至少要有一些实际代码
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            code_lines = [line for line in lines if not line.startswith('//') and line != '// Your Java code here']
            if code_lines: 
                return code
    return None

def wrap_in_solution_class(code):
    if "class Solution" in code:
        return code
    return f"class Solution {{\n{code}\n}}"

def test_generated_code(response, java_dataset_entry):
    generated_code = extract_java_code(response)
    if not generated_code:
        return False
    
    wrapped_code = wrap_in_solution_class(generated_code)
    
    imports = extract_imports_simple(java_dataset_entry["declaration"])
    test_code = java_dataset_entry['test']
    
    full_test_code = f"{imports}\n\n{wrapped_code}\n\n{test_code}"
    
    file_name = "Main.java"
    java_file_path = Path(file_name)
    
    return run_java_test(full_test_code, java_file_path)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    # TODO: load the model with quantization
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    results = []
    seed = "79244547625250131920467003550834601672"
    input_java_dataset = "selected_humanevalx_java_" + seed + ".jsonl"
    java_dataset = read_jsonl(input_java_dataset)
    
    for idx, entry in enumerate(dataset):
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        if vanilla:
            prompt = '''
You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, 
and you only answer questions related to computer science. For politically sensitive questions, 
security and privacy issues, and other non-computer science questions, you will refuse to answer.


### Instruction:
Can you translate the following Python code into Java?
The new Java code must be enclosed between [Java Start] and [Java End]
'''
            prompt += entry["prompt"] + entry["canonical_solution"]
            prompt += '''

### Response:
'''
        else:
            prompt = '''
You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, 
and you only answer questions related to computer science. For politically sensitive questions, 
security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
Translate the given Python code into Java.'''
            prompt += entry["prompt"] + entry["canonical_solution"]
            prompt += '''
The new Java code must be enclosed between [Java Start] and [Java End]
for your translated code, please also just give me the translated function code in Java, which should
be wrapped in Class Solution. What's more, for convenience for me to run test on your generated Java code, please use the delaration below:
### Java Code Declaration:
'''
            prompt += java_dataset[idx]['declaration']
            prompt +='''
Below is a response example for your reference
### Response example:
[Java Start]
class Solution {
    public static boolean ExampleJavaCode() {
        return True;
    }
}
[Java End]

''' 
        prompt += '''

### Response:
[Java Start]
'''
        # TODO: prompt the model and get the response
        input = tok(prompt, return_tensors="pt").to(m.device)
        response = m.generate(**input, temperature=0, max_new_tokens=500, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        response = tok.decode(response[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        verdict = test_generated_code(response, java_dataset[idx])

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
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
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)