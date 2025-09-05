import os
import re
import json
import sys
import git
from colorama import Fore, Style

def print_message(level="info", message=""):
    color_codes = {
        "error": Fore.RED,
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
    }

    color_code = color_codes.get(level, Fore.RESET)
    print(f"{color_code}{message}{Style.RESET_ALL}")

def extract_seed_from_selected_file(mp1_dir):
    """Extract seed value from selected_humaneval_[seed].jsonl file."""
    seed_file_pattern = r"selected_humaneval_(\d+)\.jsonl"
    for file in os.listdir(mp1_dir):
        match = re.match(seed_file_pattern, file)
        if match:
            seed_value = match.group(1)
            print_message("success", f"Extracted seed value: {seed_value} from {file}")
            return seed_value
    print_message("error", "No selected_humaneval_[seed].jsonl file found!")
    return None

def validate_repo_format(repo_url):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    pattern = r"^CS598JBR-Team-\d+$"

    if re.match(pattern, repo_name):
        print_message("success", f"Repository name {repo_name} is valid.")
        return True
    else:
        print_message(
            "error",
            f"Repository name {repo_name} is invalid."
            " It should follow the format CS598JBR-Team-[team number].",
        )
        return False

def clone_repo(repo_url):
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    if os.path.exists(repo_name):
        print(f"Repository {repo_name} already exists. Pulling latest changes...")
        repo = git.Repo(repo_name)
        repo.git.pull()
    else:
        print(f"Cloning repository {repo_name}...")
        git.Repo.clone_from(repo_url, repo_name)

    return repo_name

def validate_repo(repo_name):
    missing_files = []

    readme_file = os.path.join(repo_name, "README.md")
    if not os.path.exists(readme_file):
        print_message("error", "Validation failed: README.md is missing!")
        missing_files.append(readme_file)

    mp1_dir = os.path.join(repo_name, "MP1")
    if not os.path.exists(mp1_dir):
        print_message("error", "Validation failed: MP1 directory is missing!")
        return False

    seed_value = extract_seed_from_selected_file(mp1_dir)
    if not seed_value:
        return False

    jsonl_files_with_seed = [
        f"base_prompt_{seed_value}.jsonl",
        f"base_prompt_{seed_value}.jsonl_results.jsonl",
        f"instruct_prompt_{seed_value}.jsonl",
        f"instruct_prompt_{seed_value}.jsonl_results.jsonl",
    ]

    required_files = [
        "dataset_generation.log",
        "base_prompt.log",
        "base_evaluate.log",
        "instruct_prompt.log",
        "instruct_evaluate.log",
        "instruct_evaluate_processed.log",
        "base_evaluate_processed.log"
    ]

    all_required_files = required_files + jsonl_files_with_seed
    for file in all_required_files:
        file_path = os.path.join(mp1_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            print_message("success", f"{file} found!")

    if missing_files:
        print_message(
            "warning", f"Validation failed: Missing {', '.join(missing_files)}"
        )
        return False

    return True

def validate_jsonl_entries(file_path, expected_entries=20):
    valid_entries = 0
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f, start=1):
                try:
                    json.loads(line)
                    valid_entries += 1
                except json.JSONDecodeError:
                    print_message("error", f"Invalid JSON entry at line {i} in {file_path}.")
                    return False

        if valid_entries != expected_entries:
            print_message("error", f"{file_path} contains {valid_entries} entries, expected {expected_entries}.")
            return False
        else:
            print_message("success", f"{file_path} contains exactly {expected_entries} valid entries.")
            return True
    except Exception as e:
        print_message("error", f"Failed to validate {file_path}: {str(e)}")
        return False

def validate_jsonl_files(repo_name):
    mp1_dir = os.path.join(repo_name, "MP1")
    seed_value = extract_seed_from_selected_file(mp1_dir)
    if not seed_value:
        return False

    jsonl_files_with_seed = [
        f"base_prompt_{seed_value}.jsonl",
        f"base_prompt_{seed_value}.jsonl_results.jsonl",
        f"instruct_prompt_{seed_value}.jsonl",
        f"instruct_prompt_{seed_value}.jsonl_results.jsonl",
        f"base_prompt_processed_{seed_value}.jsonl",
        f"instruct_prompt_processed_{seed_value}.jsonl",
        f"base_prompt_processed_{seed_value}.jsonl_results.jsonl",
        f"instruct_prompt_processed_{seed_value}.jsonl_results.jsonl",
    ]

    for file in jsonl_files_with_seed:
        file_path = os.path.join(mp1_dir, file)
        if not validate_jsonl_entries(file_path):
            return False

    return True

def grade_repo(repo_url):
    if not validate_repo_format(repo_url):
        return

    repo_name = clone_repo(repo_url)
    if validate_repo(repo_name):
        print_message("success", f"Validation passed for {repo_name}!")
    else:
        print_message("error", f"Validation failed for {repo_name}!")

    if validate_jsonl_files(repo_name):
        print_message("success", f"JSONL validation passed for {repo_name}!")
    else:
        print_message("error", f"JSONL validation failed for {repo_name}!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate.py <GitHub Repo URL>")
        sys.exit(1)

    input_repo_url = sys.argv[1]
    grade_repo(input_repo_url)