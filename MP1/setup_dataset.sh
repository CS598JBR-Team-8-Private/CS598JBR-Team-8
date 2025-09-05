pip3 install datasets==2.16.1
git clone https://github.com/openai/human-eval
pip3 install -e human-eval
sed -i 's/^#\([[:space:]]*\)exec(check_program, exec_globals)/\1exec(check_program, exec_globals)/' human-eval/human_eval/execution.py
sed -i '/assert len(completion_id) == len(problems), "Some problems are not attempted."/s/^/#/' human-eval/human_eval/evaluation.py

pip3 install jsonlines
