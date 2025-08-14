import os
'''
model_name = "./results/checkpoint-20"
task = "truthfulqa_mc2"
fewshot = 0

eval_cmd = f"""
    lm_eval --model hf \
        --model_args pretrained={model_name} \
        --tasks {task} \
        --device cpu \
        --num_fewshot {fewshot}
        --limit 5
    """
os.system(eval_cmd)
'''

import lm_eval
results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=./results/checkpoint-20,trust_remote_code=True",
    tasks=["truthfulqa_mc2", "mmlu_abstract_algebra", "gsm8k"],
    device="cpu",
    num_fewshot=None, # Interate through tasks and add more shots
    limit=5,
    log_samples=True,
)
print(results["results"])






