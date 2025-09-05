import sys
import random
import hashlib
import jsonlines
from datasets import load_dataset

################################################
# Please do not change this file when doing MP3.
################################################

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def download_humanevalx_dataset(language):
    dataset = load_dataset("THUDM/humaneval-x", language)
    all_problems_output = f"humanevalx-{language}.jsonl"
    with jsonlines.open(all_problems_output, "w") as f:
        for item in dataset["test"]:
            f.write_all([item])
    print(f"Entire {language} dataset saved to {all_problems_output}")
    return dataset["test"]

def write_dataset(selected_problems, selected_problems_output, language):
    with jsonlines.open(selected_problems_output, "w") as f:
        for item in selected_problems:
            if language == "Python":
                item["task_id"] = item["task_id"].replace("Python", "HumanEval")
            elif language == "Java":
                item["task_id"] = item["task_id"].replace("Java", "HumanEval")
            f.write_all([item])
    print(f"{len(selected_problems)} saved to {selected_problems_output}")
    
def find_dataset(IDs, dataset):
    return [item for item in dataset if item["task_id"].split("/")[-1] in IDs]

def select_random_problems(mp1_humaneval_dataset, num_problems=20):
    original_dataset = read_jsonl(mp1_humaneval_dataset)
    seed = mp1_humaneval_dataset.split(".jsonl")[0].split("_")[-1]
    IDs = [item["task_id"].split("/")[-1] for item in original_dataset]
    
    python_dataset = download_humanevalx_dataset("python")
    java_dataset = download_humanevalx_dataset("java")

    selected_python_problems = find_dataset(IDs, python_dataset)
    selected_python_problems_output = f"selected_humanevalx_python_{seed}.jsonl"
    write_dataset(selected_python_problems, selected_python_problems_output, "Python")

    selected_java_promblems = find_dataset(IDs, java_dataset)
    selected_java_problems_output = f"selected_humanevalx_java_{seed}.jsonl"
    write_dataset(selected_java_promblems, selected_java_problems_output, "Java")

if __name__ == "__main__":
    args = sys.argv[1:]
    mp1_humaneval_dataset = args[0]
    select_random_problems(mp1_humaneval_dataset)

