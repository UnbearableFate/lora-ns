import os
import argparse
import json
import pdb
import jsonlines
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src import utils
import sys
from tqdm import trange
import torch._dynamo; torch._dynamo.config.suppress_errors = True
from utils.common import get_lora_rank, write_acc_to_csv, get_info_from_model_path


MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if utils.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(
    model,
    data_path,
    start=0,
    end=MAX_INT,
    batch_size=1,
    tensor_parallel_size=1,
    filepath_output=None,
    adapter_path=None,
    lora_name=None,
    lora_id=1,
    max_loras=1,
):
    if filepath_output is None:
        if adapter_path:
            filepath_output = os.path.join(adapter_path, "result_math.txt")
        else:
            filepath_output = '/'.join(model.split('/')[:-1]) + "/" + "result_math.txt"
    print(f"Result file will be dumped to {filepath_output}")
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(utils.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('length ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    max_lora_rank = get_lora_rank(adapter_path) if adapter_path else 16
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=bool(adapter_path),
        max_lora_rank=max_lora_rank,
        max_loras=max_loras,
    )
    lora_request = None
    if adapter_path:
        adapter_name = lora_name or os.path.basename(adapter_path.rstrip("/"))
        lora_request = LoRARequest(adapter_name, lora_id, lora_path=adapter_path)
    res_completions = []
    for idx in trange(len(batch_hendrycks_math_ins), desc="Predicting on MATH dataset..."):
        prompt = batch_hendrycks_math_ins[idx]
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(
            prompt,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    #print('start===', start, ', end====',end)
    #print('length====', len(results), ', acc====', acc)

    info_source = adapter_path or model
    run_info = get_info_from_model_path(info_source)
    extra_fields = {
        "timestamp": run_info.get("timestamp"),
        "seed": run_info.get("seed"),
        "extra": run_info.get("extra"),
        "model_path": os.path.basename(info_source.rstrip("/")),
        "adapter_path": adapter_path,
    }
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json") if adapter_path else None
    if adapter_cfg_path and os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        extra_fields["base_model"] = adapter_cfg.get("base_model_name_or_path")
        for key in (
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "use_dora",
            "use_rslora",
            "init_lora_weights",
        ):
            if key in adapter_cfg:
                value = adapter_cfg[key]
                extra_fields[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    write_acc_to_csv(
        filepath=filepath_output,
        eval_dataset_name="math",
        model_name=adapter_path.split("/")[-1] if adapter_path else os.path.basename(model),
        acc=acc,
        extra_fields=extra_fields,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter dir for vLLM LoRARequest (optional).",
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default=None,
        help="LoRA adapter name (defaults to adapter dir name).",
    )
    parser.add_argument(
        "--lora_id",
        type=int,
        default=1,
        help="LoRA adapter integer ID (>0).",
    )
    parser.add_argument(
        "--max_loras",
        type=int,
        default=1,
        help="Max number of LoRA adapters to load per batch.",
    )
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("--filepath_output", type=str, default="result_math.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        filepath_output=args.filepath_output,
        adapter_path=args.adapter_path,
        lora_name=args.lora_name,
        lora_id=args.lora_id,
        max_loras=args.max_loras,
    )
