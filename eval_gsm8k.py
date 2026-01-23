import os
import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import sys
from tqdm import trange
import torch._dynamo; torch._dynamo.config.suppress_errors = True
from utils.common import get_lora_rank, get_info_from_model_path, resolve_eval_csv_path, write_eval_results_csv

MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

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

def gsm8k_test(
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
    assert os.path.exists(data_path), f"{data_path} does not exist."
    assert os.path.exists(adapter_path), f"{adapter_path} does not exist."

    if filepath_output is None:
        if adapter_path:
            filepath_output = os.path.join(adapter_path, "result_gsm8k.txt")
        else:
            filepath_output = '/'.join(model.split('/')[:-1]) + "/" + "result_gsm8k.txt"
    print(f"Result file will be dumped to {filepath_output}")
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampling =====', sampling_params)
    max_lora_rank = get_lora_rank(adapter_path) 
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
    result = []
    res_completions = []
    for idx in trange(len(batch_gsm8k_ins), desc='Predicting on GSM8k'):
        prompt =  batch_gsm8k_ins[idx]
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(
            prompt,
            sampling_params,
            use_tqdm=True,
            lora_request=lora_request,
        )
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    #print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    #print('start===', start, ', end====', end)
    #print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    
    info_source = adapter_path or model
    run_info = get_info_from_model_path(info_source)
    adapter_cfg = None
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json") if adapter_path else None
    if adapter_cfg_path and os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

    base_model_name = model
    if adapter_cfg and adapter_cfg.get("base_model_name_or_path"):
        base_model_name = adapter_cfg.get("base_model_name_or_path")

    csv_path = resolve_eval_csv_path(filepath_output)
    extra_fields = {
        "adapter_path": adapter_path,
        "model_path": os.path.basename(info_source.rstrip("/")),
    }
    write_eval_results_csv(
        csv_path,
        {"accuracy": acc},
        base_model_name=base_model_name,
        dataset_name="gsm8k",
        subset="",
        timestamp=run_info.get("timestamp"),
        init_lora_weights=adapter_cfg.get("init_lora_weights") if adapter_cfg else None,
        extra=run_info.get("extra"),
        seed=run_info.get("seed"),
        peft_config=adapter_cfg,
        extra_fields=extra_fields,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
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
    parser.add_argument("--filepath_output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(
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
