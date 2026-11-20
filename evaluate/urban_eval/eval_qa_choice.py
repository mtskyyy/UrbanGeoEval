import argparse
import copy
import os
import pickle
import re

import jsonlines
import pandas as pd
import tqdm
from joblib import Parallel, delayed

from .model_client import complete

CHOICE_DIR = os.path.join("urban_data", "geoqa_choice")
RESULT_DIR = os.path.join("evaluate", "urban_eval", "results", "geoqa_choice")

GEOQA_TASK_MAPPING = {
    "node": {
        "poi2addr": "poi2addr.csv",                     # What is the address of POI (Point of Interest): xxx?
        "poi2name": "poi2name.csv",                     # What is the name of POI located at address: xxx?
        "poi2coor": "poi2coor.csv",                     # What is the longitude and latitude coordinates of xxx located in the xxx area?
    },
    "area": {
        "aoi2name": "aoi2name.csv",                     # What is the name of AOI located at address: xxx?
        "aoi2addr": "aoi2addr.csv",                     # What is the address of AOI (Area of Interest): xxx?
        "pois2aoi": "poi2aoi.csv",                      # which AOI in the xxx area do the following POIs belong to? POIs: xxx, xxx, xxx, xxx
        "poi2aoi": "pois2aoi.csv",                      # Which of the following POIs belongs to the AOI xxx located in the xxx area?
    },
    "road": {
        "road_length": "road_length.csv",               # What is the length of the road from xxx to xxx in the xxx area?
        "road__link": "road_link.csv",
        "road_junc": "road_junc.csv"
    },
    "distance": {
        "dist": "dist.csv",                             # What is the distance between "xxx" and "xxx", both located in the xxx area?
        "dist_context": "dist_context.csv",             # What is the straight-line distance between the following two locations? (Unit is km)\n1. xxx. Detailed address: xxx\n2. xxx. Detailed address: xxx
    },
    "type": {
        "poi2type": "poi2type.csv",                     # Which type do following POIs belong to? POIs: xxx, xxx, xxx, xxx
        "type2poi": "type2poi.csv",                     # Which POI belongs to type: xxx?
        "aoi2type": "aoi2type.csv",                     # Given AOI (Area of Interest) named xxx that includes the following POIs: xxx, xxx, xxx. What is the land use type for this AOI?
    },
    "function": {
        "aoi_most_poi": "aoi_most_poi.csv",             # Which of the following AOIs (Area of Interest) is likely to contain the most POIs?
        "aoi_most_area": "aoi_most_area.csv",           # Which of the following AOIs (Area of Interest) in xxx is likely to have the largest area?
        "aoi_most_poi_type": "aoi_most_poi_type.csv"    # Which of the following AOIs (Area of Interest) is likely to contain the most different types of POIs?
    },
    "surroundings": {
        "env_surround": "env_surround.csv",             # Given an environment containing several POIs: ... in the xxx area, which of the following POIs is most likely to be present in this setting?
        "poi_closest": "poi_closest.csv",               # Which of the following POIs is closest to POI xxx located in the xxx area?
    },
}

INIT_PROMPT = "The following is a multiple-choice question. Please choose the most suitable one among as the answer to this question. Please output the option directly. No need for explaination.\n"


def task_files_adaption(task_file, path_prefix):
    for category, tasks in task_file.items():
        for task_name, file_name in tasks.items():
            task_file[category][task_name] = os.path.join(path_prefix, file_name)
    os.makedirs(path_prefix, exist_ok=True)
    return task_file

def format_example(line, include_answer=False, include_answer_prompt_final=True):
    prompt = INIT_PROMPT + 'Question: ' + line['question']
    for label in 'ABCDEFGH':
        option = line.get(label)
        if pd.notna(option) and str(option).strip():
            prompt += f'\n{label}. {option}'

    if include_answer:
        prompt += f'\nAnswer: {line["answer"]}\n\n'
    elif include_answer_prompt_final:
        prompt += '\nAnswer:'
    return prompt

def extract_choice(gen):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)

    if res is None:
        res = re.search(r"\n\s*([A-H])", gen)

    if res is None:
        return "Z"

    return res.group(1)

def truncated_response(response):
    markers = ["</think>"]

    for marker in markers:
        index = response.find(marker)
        if index != -1:
            return response[index + len(marker):].strip()

    return response

def get_chat_completion(prompt, model_name, max_tokens=512, temperature=0.3,
                        print_output=True, local_vllm_port=8001):
    response = complete(
        prompt,
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        local_port=local_vllm_port,
    )
    answer = truncated_response(response)
    if print_output:
        print(f"\nModel answer: {answer}")
    return answer

def get_checkpoint_path(model_name, task_name, city_name):
    model_name = model_name.rsplit("/", 1)[-1]
    checkpoint_dir = os.path.join(RESULT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return os.path.join(checkpoint_dir, f"{city_name}_{model_name}_{task_name}_checkpoint.pkl")

def save_checkpoint(checkpoint_path, results, current_index, correct_count, count):
    checkpoint_data = {
        "results": results,
        "current_index": current_index,
        "correct_count": correct_count,
        "count": count
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return None

def get_tested_tasks(result_path):
    if os.path.exists(result_path):
        try:
            df = pd.read_csv(result_path)
            if 'task_name' in df.columns:
                return df['task_name'].tolist(), df
            return [], None
        except Exception as e:
            print(f"Could not read existing results: {e}")
            return [], None
    return [], None

def evaluate_one_case(row_data, include_answer_prompt_final, model_name, temperature, max_tokens, local_vllm_port):
    try:
        question = format_example(
            row_data,
            include_answer_prompt_final=include_answer_prompt_final,
        )

        output = get_chat_completion(
            prompt=question,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            print_output=False,
            local_vllm_port=local_vllm_port,
        )

        if output is None or len(output) == 0:
            extract = "No output"
            is_correct = False
        else:
            extract = extract_choice(output)
            is_correct = extract == row_data["answer"]

        result_item = [
            {"role": "user", "prompt": question},
            {"role": "assistant", "response": output},
            {"role": "ref", "gt": row_data["answer"]},
            {"role": "extract", "ans": extract}
        ]
        return result_item, is_correct
    except Exception as e:
        print(f"Parallel item failed: {e}")
        fallback_question = format_example(
            row_data,
            include_answer_prompt_final=include_answer_prompt_final,
        )
        fallback_item = [
            {"role": "user", "prompt": fallback_question},
            {"role": "assistant", "response": None},
            {"role": "ref", "gt": row_data["answer"]},
            {"role": "extract", "ans": "No output"}
        ]
        return fallback_item, False


def run_evaluate_api(
    task_file_path,
    model_name,
    max_validation,
    temperature,
    max_tokens,
    include_answer_prompt_final,
    city_name="New York",
    task_name="",
    workers=1,
    local_vllm_port=8001,
):
    test_df = pd.read_csv(task_file_path, header=0)

    if test_df.empty:
        print(f"Skipping empty task file: {task_file_path}")
        return [model_name, task_name, 0, 0, 0]

    if test_df.shape[0] > max_validation * 2:
        test_df = test_df.sample(max_validation, random_state=42).reset_index(drop=True)

    checkpoint_path = get_checkpoint_path(model_name, task_name, city_name)

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        print(f"Resuming from checkpoint with {checkpoint['current_index']} completed items")
        res = checkpoint["results"]
        correct_count = checkpoint["correct_count"]
        count = checkpoint["count"]
        start_index = checkpoint["current_index"] + 1
    else:
        res = []
        correct_count, count = 0, 0
        start_index = 0

    logs_dir = os.path.join(RESULT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    safe_model_name = model_name.rsplit("/", 1)[-1]
    result_path = os.path.join(logs_dir, f"{city_name}_{safe_model_name}_{task_name}.jsonl")

    use_parallel = workers > 1

    if use_parallel:
        pending_rows = [test_df.iloc[idx].to_dict() for idx in range(start_index, len(test_df))]
        if len(pending_rows) > 0:
            print(f"Running in parallel: workers={workers}, samples={len(pending_rows)}")
            parallel_results = Parallel(n_jobs=workers, backend="loky")(
                delayed(evaluate_one_case)(
                    row_data,
                    include_answer_prompt_final,
                    model_name,
                    temperature,
                    max_tokens,
                    local_vllm_port,
                )
                for row_data in tqdm.tqdm(pending_rows, total=len(pending_rows))
            )

            with jsonlines.open(result_path, mode="a") as f:
                for result_item, is_correct in parallel_results:
                    res.append(result_item)
                    f.write(result_item)
                    if is_correct:
                        correct_count += 1
                    count += 1

            save_checkpoint(checkpoint_path, res, len(test_df) - 1, correct_count, count)
    else:
        for idx in tqdm.tqdm(range(start_index, len(test_df)), initial=start_index, total=len(test_df)):
            try:
                row = test_df.iloc[idx]

                question = format_example(
                    row,
                    include_answer_prompt_final=include_answer_prompt_final,
                )

                output = get_chat_completion(
                    prompt=question,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    local_vllm_port=local_vllm_port,
                )

                if output is None or len(output) == 0:
                    extract = "No output"
                else:
                    extract = extract_choice(output)

                result_item = [
                    {"role":"user", "prompt": question},
                    {"role":"assistant", "response": output},
                    {"role":"ref", "gt": row["answer"]},
                    {"role":"extract", "ans": extract}
                ]
                res.append(result_item)

                with jsonlines.open(result_path, mode="a") as f:
                    f.write(result_item)

                if output is not None and len(output) > 0:
                    ans = extract_choice(output)
                    if ans == row["answer"]:
                        correct_count += 1
                count += 1

                save_checkpoint(checkpoint_path, res, idx, correct_count, count)

            except Exception as e:
                print(f"Item {idx} failed: {e}")
                save_checkpoint(checkpoint_path, res, idx-1, correct_count, count)
                continue

    accuracy = correct_count / count if count > 0 else 0
    print("Success rate:{}({}/{})".format(accuracy, correct_count, count))

    return [model_name, task_name, correct_count, count, accuracy]


if __name__ == "__main__":

    supported_model_list = [
        'qwen2.5-7b-instruct',
        'qwen2.5-72b-instruct',
        'qwen3.5-397b-a17b'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, nargs='+', default=["New York"], help="Space- or comma-separated cities, or 'all'.")
    parser.add_argument('--model_name', type=str, nargs='+', required=True, help="Model IDs served by the configured endpoint.")
    parser.add_argument("--max_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.3, type=float)
    parser.add_argument("--max_valid", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--local_vllm_port", type=int, default=8001, help="Local endpoint port when no base URL is configured.")
    parser.add_argument("--include_answer_prompt_final", action="store_true", help="Append an answer cue to the prompt.") # adds an "Answer" suffix at the end of the prompt
    args = parser.parse_args()

    def resolve_city_list(city_inputs, city_root=CHOICE_DIR):
        parsed_cities = []
        for item in city_inputs:
            parsed_cities.extend([x.strip() for x in item.split(",") if x.strip()])

        parsed_cities_lower = [city.lower() for city in parsed_cities]
        if "all" in parsed_cities_lower:
            all_cities = []
            for city in os.listdir(city_root):
                city_path = os.path.join(city_root, city)
                if os.path.isdir(city_path):
                    all_cities.append(city)
            return sorted(all_cities)

        unique_cities = []
        for city in parsed_cities:
            if city not in unique_cities:
                unique_cities.append(city)
        return unique_cities

    city_list = resolve_city_list(args.city_name)

    if len(city_list) == 0:
        raise ValueError("No evaluation cities were selected.")

    print("Evaluating city list:", city_list)
    print("Evaluating model list: ", args.model_name)

    model_eval_list = supported_model_list if args.model_name[0] == "all" else args.model_name

    for city_name in city_list:
        task_path = os.path.join(CHOICE_DIR, city_name)
        if not os.path.exists(task_path):
            print(f"Skipping {city_name}: data directory not found at {task_path}")
            continue

        print(f"\n==================== Evaluating {city_name} ====================")
        NEW_TASK_FILES = task_files_adaption(copy.deepcopy(GEOQA_TASK_MAPPING), task_path)

        for model in model_eval_list:
            safe_model_name = model.rsplit("/", 1)[-1]
            result_path = os.path.join(RESULT_DIR, f"{city_name}_{safe_model_name}.csv")
            tested_tasks, existing_results = get_tested_tasks(result_path)

            res_df = []

            for category, tasks in NEW_TASK_FILES.items():
                for task_name, file_path in tasks.items():
                    if task_name in tested_tasks:
                        print(f"Skipping a completed task: {task_name}")
                        continue

                    print(f"Evaluating {model} on {file_path}")
                    res = run_evaluate_api(
                        file_path,
                        model,
                        args.max_valid,
                        args.temperature,
                        args.max_tokens,
                        include_answer_prompt_final=args.include_answer_prompt_final,
                        city_name=city_name,
                        task_name=task_name,
                        workers=args.workers,
                        local_vllm_port=args.local_vllm_port,
                    )
                    res_df.append(res)

            if res_df:
                os.makedirs(RESULT_DIR, exist_ok=True)

                columns = ["model_name", "task_name", "corrct", "count", "accuracy"]
                if existing_results is None:
                    result_df = pd.DataFrame(res_df, columns=columns)
                else:
                    new_rows = pd.DataFrame(res_df, columns=columns)
                    result_df = pd.concat([existing_results, new_rows], ignore_index=True)

                result_df.to_csv(result_path, index=False)
            else:
                print(f"No new tasks for {model} in {city_name}")
