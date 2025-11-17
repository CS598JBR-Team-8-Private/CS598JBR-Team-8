import jsonlines
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

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
        if vanilla:
            prompt = '''
You are an AI programming assistant. You are an AI programming assistant utilizing the DeepSeek Coder 
model, developed by DeepSeek Company, and you only answer questions related to computer science. 
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:           
            
'''
            prompt += entry["declaration"] + entry["buggy_solution"]
            prompt += '''
Is the above code buggy or correct? Please explain your step by step reasoning. The prediction should 
be enclosed within <start> and <end> tags. For example: <start>Buggy<end>

### Response:
'''
        else:
            prompt = '''
You are an AI programming assistant. You are an AI programming assistant utilizing the DeepSeek Coder 
model, developed by DeepSeek Company, and you only answer questions related to computer science. 
For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will 
refuse to answer.

### Instruction:           
            
'''
            prompt += entry["declaration"] + entry["buggy_solution"] + entry["test"]
            prompt += '''
Is the above code buggy or correct? Please explain your step by step reasoning. Please notice that here the code given consists
of two parts, function and test, your goal is to judge whether the given function can pass the test successfully, if yes, then
the code is correct, else the code is buggy.
The prediction should  be enclosed within <start> and <end> tags. For example: <start>Buggy<end>
'''
            
            prompt +='''
### Response:
'''
        # TODO: prompt the model and get the response
        input = tok(prompt, return_tensors="pt").to(m.device)
        response = m.generate(**input, temperature=0, max_new_tokens=500)
        response = tok.decode(response[0], skip_special_tokens=True)

        # TODO: process the response and save it to results
        pattern = r"<start>(.*?)<end>"

        # re.findall() returns a list of all substrings that matched the capturing group (.*?)
        matches = re.findall(pattern, response, re.DOTALL)
        verdict = False
        if len(matches) >= 3:
            answer = matches[-1]
        
            if answer == "Buggy":
                verdict = True
        

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
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
    This Python script is to run prompt LLMs for bug detection.
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
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
