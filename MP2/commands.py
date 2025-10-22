###################################################################
# This is a list of all commands you need to run for MP2 on Colab.
###################################################################

# TODO: Clone your GitHub repository
! git clone https://github.com/CS598JBR-Team-8-Private/CS598JBR-Team-8.git
% cd CS598JBR-Team-8/MP2

# TODO: Replace the file path of selected_humaneval_[seed].jsonl generated in MP1
input_dataset = "selected_humaneval_79244547625250131920467003550834601672.jsonl"# selected_humaneval_[seed].jsonl

# Set up requirements for model prompting
! bash -x setup_models.sh

# TODO: add your seed generated in MP1
seed = "79244547625250131920467003550834601672"
task_1_vanilla_json = "task_1_" + seed + "_vanilla.jsonl"
task_1_crafted_json = "task_1_" + seed + "_crafted.jsonl"
task_2_vanilla_json = "task_2_" + seed + "_vanilla.jsonl"
task_2_crafted_json = "task_2_" + seed + "_crafted.jsonl"

# Prompt the models, you can modify `MP2/task_1.py, MP2/task_2.py`
# The {input_dataset} is the JSON file consisting of 20 unique programs for your group that you generated in MP1 (selected_humaneval_[seed].jsonl)
! python3 task_1.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_1_vanilla_json} "True" |& tee task_1_vanilla.log
! python3 task_1.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_1_crafted_json} "False" |& tee task_1_crafted.log
! python3 task_2.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_2_vanilla_json} "True" |& tee task_2_vanilla.log
! python3 task_2.py {input_dataset} "deepseek-ai/deepseek-coder-6.7b-instruct" {task_2_crafted_json} "False" |& tee task_2_crafted.log

# Commands to generate coverage reports

%cd ..

# git push all nessacery files (e.g., *jsonl, *log) to your GitHub repository
