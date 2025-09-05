import jsonlines
import os
import re
import sys
from datasets import load_dataset

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset

def write_jsonl(file_path, dataset):
    with jsonlines.open(file_path, mode='w') as writer:
        for entry in dataset:
            writer.write(entry)

def convert_to_humanevalpack(input_file, output_file):
    humanevalpack_data = load_dataset("bigcode/humanevalpack", "python", trust_remote_code=True)["test"]
    dataset = read_jsonl(input_file)    
    converted_dataset = []

    for entry in dataset:
        task_id = entry["task_id"]
        task_id = task_id.replace("HumanEval", "Python")

        new_entry = next((item for item in humanevalpack_data if item["task_id"] == task_id), None)
        
        if new_entry is None:
            print(f"Task_ID {task_id} not found in humanevalpack_data")
            continue

        new_entry["task_id"] = entry["task_id"]
        converted_dataset.append(new_entry)

    write_jsonl(output_file, converted_dataset)
    print(f"Conversion completed. Output saved to {output_file}")

def extract_seed_from_filename(file_name):
    match = re.search(r'\d+', file_name)
    return match.group(0) if match else None

if __name__ == "__main__":
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    seed = extract_seed_from_filename(input_file)
    output_file = "selected_humanevalpack_" + seed + ".jsonl"
    convert_to_humanevalpack(input_file, output_file)
