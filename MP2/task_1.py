import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import re
import ast

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def extract_test(test_string, seed=598):
    pattern = r'''assert\s+candidate\((.*?)\)\s*==\s*(.*?)(?=(?:,\s*['"][^'"]*['"])?\s*$)'''
    assertions = re.findall(pattern, test_string, re.MULTILINE | re.VERBOSE)

    if not assertions:
        return None

    if seed is not None:
        random.seed(seed)
    selected_assertion = random.choice(assertions)

    input_value_str = selected_assertion[0].strip()
    expected_output_str = selected_assertion[1].strip()

    return input_value_str, expected_output_str

def extract_output_content(response: str) -> str:
    pattern = r"\[Output\](.*?)\[/Output\]"
    all_matches = re.findall(pattern, response, re.DOTALL)

    if all_matches:
        return all_matches[-1].strip()
    else:
        return ""

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

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
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        result = extract_test(entry['test'])
        if result is not None:
            input_value, expected_output = result
        else:
            input_value = expected_output = ""

        if vanilla:
            prompt = """"You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."""
            prompt += f"### Instruction:\n If the input is {input_value}, what will the following code return?"

            prompt += """The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]"""
            prompt += entry["prompt"] + entry["canonical_solution"]
        else:
            prompt = """"You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."""
            prompt += f'''### Instruction:\n If the input is {input_value}, what will the following code return? Reason step by step to 
                solve the problem. What's more, for the response in the prediction, please do not contain any other characters or blank space, 
                just the predicted returned value of the function given. Also for the response, only contain exact one pair of the [Output] and [/Output]
                tages, and the predicted returned value of the function should be in between the two tags. Here is an example:\n
                ###Response
                For the string '[]]]]]]][[[[[]', the function will first identify the indices of the opening and closing brackets as follows:
                opening_bracket_index = []
                closing_bracket_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                Then it will check if the indices of the closing brackets are greater than the indices of the opening brackets. The result will be:
                False, False, False, False, False, False, False, False, False, False, False, False
                Since there are no True values in the result, the function will return False.
                So, the return value of the function is [Output]False[/Output].'''
            prompt += """The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]"""
            prompt += entry["prompt"] + entry["canonical_solution"]
          
        # TODO: prompt the model and get the response
        input = tok(prompt, return_tensors="pt").to(m.device)
        response = m.generate(**input, temperature=0, max_new_tokens=500)
        response = tok.decode(response[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        verdict = False
        pred_answer = extract_output_content(response)
        if str(pred_answer) == expected_output:
          verdict = True

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
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
