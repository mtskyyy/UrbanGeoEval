import argparse
import concurrent.futures
import glob
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

if __package__:
    from .model_client import complete
    from .reason_prompt import CHECKLISTS, INIT_PROMPT, SCORING_PROMPT, TASK_PROMPT_LIST
else:
    from model_client import complete
    from reason_prompt import CHECKLISTS, INIT_PROMPT, SCORING_PROMPT, TASK_PROMPT_LIST

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


REASON_DIR = Path("urban_data/geoqa_reason")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_FILE_NAME = MODEL_NAME.rsplit("/", 1)[-1]
RESULT_DIR = Path("evaluate/urban_eval/results/geoqa_reason") / MODEL_FILE_NAME
PROGRAM_TASKS = {"distance", "direction", "planning"}

file_lock = threading.Lock()


def truncated_response(response):
    markers = ["<reason>", "<reasoning>",
               "<reasoning_process>", "<reasoning process>"]

    for marker in markers:
        index = response.find(marker)
        if index != -1:
            return response[index:].strip()

    return response


def query_llm(prompt):
    start = time.perf_counter()
    try:
        response = complete(prompt, MODEL_NAME, temperature=0.6)
    except Exception as exc:
        logger.error("Could not query %s: %s", MODEL_NAME, exc)
        return None, 0, 0

    answer = truncated_response(response.rsplit("</think>", 1)[-1])
    return answer, len(response), time.perf_counter() - start


def score_with_judge(task_name, question, response, ground_truth, args):
    """Score the reasoning process and the final answer with a configurable judge."""
    checklist = CHECKLISTS[task_name]
    prompt = (
        f"{SCORING_PROMPT}\n\nQuestion:\n{question}\n\n"
        f"Model's Response:\n{response}\n\nChecklist:\n{checklist}\n\n"
        f"Ground Truth:\n{ground_truth}"
    )
    scores = {"reason_score": None, "answer_score": None, "total_score": None}
    try:
        evaluation = complete(
            prompt, args.judge_model, temperature=0.0, max_tokens=3000
        ).rsplit("</think>", 1)[-1]
    except Exception as exc:
        logger.error("Judge request failed: %s", exc)
        return None, scores

    for label in ("reason", "answer", "total"):
        match = re.search(rf"{label}:\s*\[\[(\d+(?:\.\d+)?)\]\]", evaluation, re.IGNORECASE)
        if match:
            scores[f"{label}_score"] = float(match.group(1))
    if any(value is None for value in scores.values()):
        logger.warning("Judge did not return all scores for %s", task_name)
    return evaluation, scores


def build_task_prompt(question, task_name):
    """Build the prompt for the given question."""
    if "distance" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['distance'] + "\n\n" + "Question: " + question
    elif "direction" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['direction'] + "\n\n" + "Question: " + question
    elif "relation" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['relation'] + "\n\n" + "Question: " + question
    elif "planning" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['planning'] + "\n\n" + "Question: " + question
    elif "prediction" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['prediction'] + "\n\n" + "Question: " + question
    elif "navigation" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['navigation'] + "\n\n" + "Question: " + question
    elif "recommendation" in task_name:
        return INIT_PROMPT + "\n\n" + TASK_PROMPT_LIST['recommendation'] + "\n\n" + "Question: " + question
    else:
        print("Error: task name not supported")
        return INIT_PROMPT + "\n\n" + question


def calculate_average_scores():
    """
    Calculate average scores for each task and save them to a CSV file.

    1. Reads all result JSONL files
    2. Calculates average reason, answer, and total scores for each task
    """
    logger.info(f"Calculating average scores for {MODEL_NAME}...")

    # Get all result files of the current model
    result_files = glob.glob(os.path.join(RESULT_DIR, f"*_{MODEL_FILE_NAME}.jsonl"))

    if not result_files:
        logger.warning(f"No result files found for {MODEL_NAME}")
        return

    # Create a DataFrame to store the average scores
    scores_data = []

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        task_name = file_name.split('_')[0]

        reason_scores = []
        answer_scores = []
        total_scores = []

        # Read the JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry["reason_score"] is not None:
                    reason_scores.append(entry["reason_score"])
                if entry["answer_score"] is not None:
                    answer_scores.append(entry["answer_score"])
                if entry["total_score"] is not None:
                    total_scores.append(entry["total_score"])

        # Calculate averages
        if reason_scores:
            reason_avg = sum(reason_scores) / len(reason_scores)
        else:
            reason_avg = None

        if answer_scores:
            answer_avg = sum(answer_scores) / len(answer_scores)
        else:
            answer_avg = None

        if total_scores:
            total_avg = sum(total_scores) / len(total_scores)
        else:
            total_avg = None

        # Add to scores data
        scores_data.append({
            "model_name": MODEL_NAME,
            "task_name": task_name,
            "reason_avg": round(reason_avg, 2) if reason_avg is not None else None,
            "answer_avg": round(answer_avg, 2) if answer_avg is not None else None,
            "total_avg": round(total_avg, 2) if total_avg is not None else None
        })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(scores_data)
    df = df.sort_values(by=['task_name'])

    # Display the results in a table format
    logger.info("\nAverage Scores:")
    logger.info(df.to_string(index=False))

    return df


def process_single_case(entry, task_name, args):
    """Generate and score one response."""
    id = entry["id"]
    question = entry["question"]
    ground_truth = entry["gt"]
    prompt = build_task_prompt(question, task_name)

    # Query the model
    golden_solution, length, time_used = query_llm(prompt)

    time_used = round(time_used, 2)  # round to 2 decimal places

    if golden_solution is None or len(golden_solution) == 0:
        logger.error(
            f"Error: model {MODEL_NAME} response is None or empty for id {id}, skipping this line")
        return None

    # Score the response with the configured judge.
    evaluation, scores = score_with_judge(
        task_name, question, golden_solution, ground_truth, args)

    # Create result entry
    result_entry = {
        "id": id,
        "response": golden_solution,
        "ground_truth": ground_truth,
        "text_len": length,
        "time_used": time_used,
        "evaluation": evaluation,
        "reason_score": scores["reason_score"],
        "answer_score": scores["answer_score"],
        "total_score": scores["total_score"]
    }

    print(
        f"\nID: {id}\n#GT#: {ground_truth}\n\nreason: {scores['reason_score']}\nanswer: {scores['answer_score']}\ntotal: {scores['total_score']}\ntext len: {length}\ntime_used: {time_used} sec\n")

    return result_entry


def process_jsonl_file(file_path, args):
    """Evaluate the remaining cases in one task file."""
    task_name = Path(file_path).stem.removesuffix("_case")
    result_file = os.path.join(RESULT_DIR, f"{task_name}_{MODEL_FILE_NAME}.jsonl")

    existing_results = {}
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'id' in entry:
                        existing_results[entry['id']] = True
                except json.JSONDecodeError:
                    continue

    else:
        with open(result_file, 'w', encoding='utf-8') as f:
            pass

    with open(file_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    lines_to_process = [case for case in cases if case["id"] not in existing_results]

    print(
        f"Evaluating task: {task_name}... Found {len(lines_to_process)} new cases to process.")

    if not lines_to_process:
        logger.info(f"No new records to process for task {task_name}.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(
            process_single_case, entry, task_name, args): entry['id'] for entry in lines_to_process}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines_to_process), desc=f"Processing {task_name}"):
            result_entry = future.result()

            if result_entry:
                with file_lock:
                    with open(result_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_entry) + '\n')

    logger.info(
        f"Completed processing {task_name}, results saved to {result_file}")


def write_results(path, records):
    """Replace a result file only after every record has been written."""
    destination = Path(path)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        with staged.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        staged.replace(destination)


def reassess_null_scores(args):
    """Retry records with incomplete judge scores."""
    result_files = glob.glob(os.path.join(RESULT_DIR, "*.jsonl"))

    if not result_files:
        logger.warning("No result files found")
        return

    logger.info(f"Found {len(result_files)} result files to process")

    total_fixed = 0

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        task_name = file_name.split('_')[0]
        model_name = '_'.join(file_name.split('_')[1:]).replace('.jsonl', '')

        logger.info(f"Processing {file_name}")

        original_case_file = REASON_DIR / f"{task_name}_case.jsonl"
        if not os.path.exists(original_case_file):
            logger.warning(
                f"Original case file not found: {original_case_file}")
            continue

        id_to_question = {}
        id_to_gt = {}
        with open(original_case_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    id_to_question[case.get('id')] = case.get('question', '')
                    id_to_gt[case.get('id')] = case.get('gt', '')
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse a line in {original_case_file}")

        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse a line in {file_path}")

        records_to_fix = []
        for i, record in enumerate(records):
            if (record.get('reason_score') is None or
                record.get('answer_score') is None or
                    record.get('total_score') is None):
                records_to_fix.append((i, record))

        if not records_to_fix:
            logger.info(f"No records with null scores found in {file_name}")
            continue

        logger.info(
            f"Found {len(records_to_fix)} records with null scores in {file_name}")

        fixed_count = 0
        for idx, record in tqdm(records_to_fix, desc=f"Reassessing {file_name}"):
            record_id = record.get('id')
            if not record_id:
                logger.warning(
                    f"Record at index {idx} does not have an ID. Skipping...")
                continue

            question = id_to_question.get(record_id, '')
            ground_truth = id_to_gt.get(record_id, '')

            if not question:
                logger.warning(
                    f"Question not found for record ID {record_id}. Skipping...")
                continue

            response = record.get('response', '')

            print(
                f"Reassess Task: {task_name} for Model: {model_name} on record ID: {record_id}")

            if "</think>" in response:
                response = response.split("</think>")[-1]

            if len(response) > 30000:
                print(
                    f"warn: model response length is too long ({len(response)}), cut to last 30000")
                response = response[-30000:]

            print(response + "\n\n")
            evaluation, scores = score_with_judge(
                task_name, question, response, ground_truth, args)

            if any(score is None for score in scores.values()):
                continue

            records[idx]['evaluation'] = evaluation
            records[idx]['reason_score'] = scores['reason_score']
            records[idx]['answer_score'] = scores['answer_score']
            records[idx]['total_score'] = scores['total_score']

            if all(score is not None for score in scores.values()):
                fixed_count += 1
                logger.info(f"Successfully reassessed record ID {record_id}:")
                logger.info(
                    f"  Reason: {scores['reason_score']}, Answer: {scores['answer_score']}, Total: {scores['total_score']}")

        total_fixed += fixed_count

        if fixed_count > 0:
            logger.info(
                f"Successfully reassessed {fixed_count} out of {len(records_to_fix)} records in {file_name}")

            write_results(file_path, records)

            logger.info(f"Updated records saved to {file_path}")

    logger.info(f"Reassessment completed. Total records fixed: {total_fixed}")


def reassess_single_record_with_overwrite(record, task_name, id_to_question, id_to_gt, args):
    """Rejudge one saved response."""
    record_id = record.get('id')
    if not record_id:
        logger.warning(f"Record does not have an ID. Skipping...")
        return None

    question = id_to_question.get(record_id, '')
    ground_truth = id_to_gt.get(record_id, '')

    if not question:
        logger.warning(
            f"Question not found for record ID {record_id}. Skipping...")
        return None

    response = record.get('response', '')

    if "</think>" in response:
        response = response.split("</think>")[-1]

    if len(response) > 30000:
        logger.warning(
            f"Model response length is too long ({len(response)}), cut to last 30000")
        response = response[-30000:]

    evaluation, scores = score_with_judge(
        task_name, question, response, ground_truth, args)

    if any(score is None for score in scores.values()):
        return None

    updated_record = record.copy()
    updated_record['evaluation'] = evaluation
    updated_record['reason_score'] = scores['reason_score']
    updated_record['answer_score'] = scores['answer_score']
    updated_record['total_score'] = scores['total_score']

    return updated_record


def reassess_all_records_with_overwrite(args):
    """Rejudge all saved responses for the current model."""
    result_files = glob.glob(os.path.join(RESULT_DIR, "*.jsonl"))

    if not result_files:
        logger.warning("No result files found")
        return

    logger.info(
        f"Found {len(result_files)} result files to process with overwrite mode")
    logger.info(f"Using {args.workers} workers for parallel processing")

    total_processed = 0

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        task_name = file_name.split('_')[0]
        model_name = '_'.join(file_name.split('_')[1:]).replace('.jsonl', '')

        logger.info(f"Processing {file_name} with overwrite")

        original_case_file = REASON_DIR / f"{task_name}_case.jsonl"
        if not os.path.exists(original_case_file):
            logger.warning(
                f"Original case file not found: {original_case_file}")
            continue

        id_to_question = {}
        id_to_gt = {}
        with open(original_case_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    id_to_question[case.get('id')] = case.get('question', '')
                    id_to_gt[case.get('id')] = case.get('gt', '')
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse a line in {original_case_file}")

        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse a line in {file_path}")

        if not records:
            logger.info(f"No records found in {file_name}")
            continue

        logger.info(f"Reassessing all {len(records)} records in {file_name}")

        updated_records = Parallel(n_jobs=args.workers, backend='threading')(
            delayed(reassess_single_record_with_overwrite)(
                record, task_name, id_to_question, id_to_gt, args
            )
            for record in tqdm(records, desc=f"Reassessing {file_name}")
        )

        valid_records = [r for r in updated_records if r is not None]
        processed_count = len(valid_records)
        total_processed += processed_count

        logger.info(
            f"Successfully reassessed {processed_count} out of {len(records)} records in {file_name}")

        if valid_records:
            write_results(file_path, (
                updated if updated is not None else original
                for original, updated in zip(records, updated_records)
            ))

            logger.info(f"Updated records saved to {file_path}")

    logger.info(
        f"Overwrite reassessment completed. Total records processed: {total_processed}")


def extract_value_with_llm(response, task_name, args):
    """Extract answer values from a saved response using the judge."""
    extraction_prompts = {
        "distance": """Extract the distance value and unit from the model's response <answer> tag.

Output in JSON format:
{{
    "response_distance": <value>,
    "response_unit": "<unit>"
}}

Response: {response}""",

        "direction": """Extract the bearing value (in degrees) and cardinal direction from the model's response <answer> tag.

Output in JSON format:
{{
    "response_bearing": <value>,
    "response_direction": "<direction>"
}}

Response: {response}""",

        "planning": """Extract the visiting order as a list of location names from the model's response <answer> tag.
The locations should be in the exact sequence mentioned, separated by "->".

Output in JSON format:
{{
    "response_order": ["LocationA", "LocationB", "LocationC", ...]
}}

Response: {response}"""
    }

    if task_name not in extraction_prompts:
        logger.warning(f"Task {task_name} does not support program evaluation")
        return None

    prompt = extraction_prompts[task_name].format(response=response[:30000])

    try:
        llm_response = complete(
            prompt, args.judge_model, temperature=0.0, max_tokens=2048
        ).rsplit("</think>", 1)[-1]

        if not llm_response:
            logger.error("LLM failed to extract values")
            return None

        json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
        if json_match:
            extracted_data = json.loads(json_match.group())
            return extracted_data
        else:
            logger.warning(
                f"Could not find JSON in LLM response: {llm_response}")
            return None

    except Exception as e:
        logger.error(f"Error extracting values with LLM: {e}")
        return None


def calculate_distance_score(ground_truth_distance_km, response_distance_km):
    """Score the absolute distance error in kilometers."""
    error = abs(response_distance_km - ground_truth_distance_km)
    score = max(0, 10 - error)

    return round(score, 2)


def calculate_direction_score(ground_truth_bearing, response_bearing):
    """Score the circular error between two bearings."""
    error = abs(response_bearing - ground_truth_bearing)
    if error > 180:
        error = 360 - error

    score = max(0, 10 - error * 0.25)

    return round(score, 2)


def levenshtein_distance(seq1, seq2):
    """Return the edit distance between two sequences."""
    m, n = len(seq1), len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )

    return dp[m][n]


def calculate_planning_score(ground_truth_order, response_order):
    """Score the visiting order using sequence edit distance."""
    def normalize_location(loc):
        return loc.strip().lower()

    gt_normalized = [normalize_location(loc) for loc in ground_truth_order]
    resp_normalized = [normalize_location(loc) for loc in response_order]

    edit_distance = levenshtein_distance(gt_normalized, resp_normalized)

    penalty = edit_distance * 2
    score = max(0, 10 - penalty)

    correct_positions = 0
    for i in range(min(len(gt_normalized), len(resp_normalized))):
        if gt_normalized[i] == resp_normalized[i]:
            correct_positions += 1

    return round(score, 2), {
        "edit_distance": edit_distance,
        "correct_positions": correct_positions,
        "total_positions": len(gt_normalized),
        "ground_truth_order": ground_truth_order,
        "response_order": response_order
    }


def program_evaluate_single_record(record, task_name, id_to_gt, args):
    """Recompute the answer score for one saved response."""
    record_id = record.get('id')
    if not record_id:
        logger.warning("Record does not have an ID. Skipping...")
        return None

    if id_to_gt:
        ground_truth = id_to_gt.get(record_id, '')
    else:
        ground_truth = record.get('ground_truth', '')

    if not ground_truth:
        logger.warning(
            f"Ground truth not found for record ID {record_id}. Skipping...")
        return None

    response = record.get('response', '')
    if "</think>" in response:
        response = response.split("</think>")[-1]

    extracted_data = extract_value_with_llm(response, task_name, args)

    if not extracted_data:
        logger.warning(f"Failed to extract data for record ID {record_id}")
        return None

    answer_score = None
    program_eval_details = {}

    try:
        if task_name == "distance":
            response_distance = extracted_data.get('response_distance')
            response_unit = extracted_data.get('response_unit', 'km')

            if response_distance is not None:
                if 'km' in response_unit.lower():
                    response_distance_km = float(response_distance)
                elif 'm' in response_unit.lower() and 'km' not in response_unit.lower():
                    response_distance_km = float(response_distance) / 1000
                else:
                    response_distance_km = float(response_distance)

                gt_match = re.search(r'([\d.]+)\s*(km|m)', ground_truth.lower())
                if gt_match:
                    gt_distance = float(gt_match.group(1))
                    gt_unit = gt_match.group(2)

                    if gt_unit == 'km':
                        ground_truth_distance_km = gt_distance
                    elif gt_unit == 'm':
                        ground_truth_distance_km = gt_distance / 1000
                    else:
                        ground_truth_distance_km = gt_distance

                    answer_score = calculate_distance_score(
                        ground_truth_distance_km, response_distance_km
                    )
                    program_eval_details = {
                        "ground_truth_distance_km": round(ground_truth_distance_km, 2),
                        "model_distance_km": round(response_distance_km, 2),
                        "error_km": round(abs(response_distance_km - ground_truth_distance_km), 2)
                    }

        elif task_name == "direction":
            response_bearing = extracted_data.get('response_bearing')

            if response_bearing is not None:
                gt_match = re.search(r'([\d.]+)\s*(?:°|degrees?)', ground_truth.lower())
                if gt_match:
                    ground_truth_bearing = float(gt_match.group(1))

                    answer_score = calculate_direction_score(
                        ground_truth_bearing, float(response_bearing)
                    )
                    program_eval_details = {
                        "ground_truth_bearing_deg": round(ground_truth_bearing, 2),
                        "model_bearing_deg": round(float(response_bearing), 2),
                        "error_deg": round(abs(float(response_bearing) - ground_truth_bearing), 2)
                    }

        elif task_name == "planning":
            response_order = extracted_data.get('response_order', [])

            if '->' in ground_truth:
                ground_truth_order = [loc.strip()
                                      for loc in ground_truth.split('->')]
            else:
                ground_truth_order = [
                    ground_truth.strip()] if ground_truth else []

            if response_order and ground_truth_order:
                answer_score, details = calculate_planning_score(
                    ground_truth_order, response_order
                )
                program_eval_details = details

    except Exception as e:
        logger.error(f"Error calculating score for record ID {record_id}: {e}")
        return None

    if answer_score is None:
        logger.warning(
            f"Could not calculate answer score for record ID {record_id}")
        return None

    reason_score = record.get('reason_score', 0)
    if reason_score is None:
        reason_score = 0

    total_score = reason_score + answer_score

    updated_record = record.copy()
    updated_record['answer_score'] = answer_score
    updated_record['total_score'] = round(total_score, 2)
    updated_record['program_eval_details'] = program_eval_details
    updated_record['evaluation_method'] = 'program'

    return updated_record


def calculate_score_correlation(original_scores, program_scores, task_name):
    """Compare the original judge scores with programmatic scores."""
    if len(original_scores) != len(program_scores) or len(original_scores) == 0:
        logger.warning(f"Task {task_name}: Cannot calculate correlation - invalid data length")
        return None

    valid_pairs = [(orig, prog) for orig, prog in zip(original_scores, program_scores)
                   if orig is not None and prog is not None]

    if len(valid_pairs) < 2:
        logger.warning(f"Task {task_name}: Not enough valid data points for correlation")
        return None

    original_valid = [pair[0] for pair in valid_pairs]
    program_valid = [pair[1] for pair in valid_pairs]

    pearson_corr, pearson_pvalue = stats.pearsonr(original_valid, program_valid)

    spearman_corr, spearman_pvalue = stats.spearmanr(original_valid, program_valid)

    mae = np.mean(np.abs(np.array(original_valid) - np.array(program_valid)))
    rmse = np.sqrt(np.mean((np.array(original_valid) - np.array(program_valid))**2))

    return {
        "task_name": task_name,
        "n_samples": len(valid_pairs),
        "pearson_r": round(pearson_corr, 4),
        "pearson_pvalue": round(pearson_pvalue, 4),
        "spearman_r": round(spearman_corr, 4),
        "spearman_pvalue": round(spearman_pvalue, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "original_mean": round(np.mean(original_valid), 2),
        "original_std": round(np.std(original_valid), 2),
        "program_mean": round(np.mean(program_valid), 2),
        "program_std": round(np.std(program_valid), 2)
    }


def program_evaluate_all_records(args):
    """Recompute answer scores and preserve the existing reasoning scores."""
    result_files = glob.glob(os.path.join(RESULT_DIR, "*.jsonl"))

    if not result_files:
        logger.warning("No result files found")
        return

    logger.info(
        f"Found {len(result_files)} result files for program evaluation")
    logger.info(f"Using {args.workers} workers for parallel processing")

    total_processed = 0
    correlation_results = []

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        task_name = file_name.split('_')[0]

        if task_name not in PROGRAM_TASKS:
            logger.info(
                f"Skipping {file_name} - task does not support program evaluation")
            continue

        logger.info(f"Processing {file_name} with program evaluation")

        original_case_file = REASON_DIR / f"{task_name}_case.jsonl"
        if not os.path.exists(original_case_file):
            logger.warning(
                f"Original case file not found: {original_case_file}")
            continue

        id_to_gt = {}
        with open(original_case_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    id_to_gt[case.get('id')] = case.get('gt', '')
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse a line in {original_case_file}")

        records = []
        original_answer_scores = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                    original_answer_scores.append(record.get('answer_score'))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse a line in {file_path}")

        if not records:
            logger.info(f"No records found in {file_name}")
            continue

        logger.info(
            f"Program evaluating all {len(records)} records in {file_name}")

        updated_records = Parallel(n_jobs=args.workers, backend='threading')(
            delayed(program_evaluate_single_record)(
                record, task_name, id_to_gt, args
            )
            for record in tqdm(records, desc=f"Program evaluating {file_name}")
        )

        valid_records = []
        program_answer_scores = []
        original_scores_filtered = []

        for i, updated_record in enumerate(updated_records):
            if updated_record is not None:
                valid_records.append(updated_record)
                program_answer_scores.append(updated_record.get('answer_score'))
                original_scores_filtered.append(original_answer_scores[i])

        processed_count = len(valid_records)
        total_processed += processed_count

        logger.info(
            f"Successfully program evaluated {processed_count} out of {len(records)} records in {file_name}")

        correlation_result = calculate_score_correlation(
            original_scores_filtered, program_answer_scores, task_name
        )
        if correlation_result:
            correlation_results.append(correlation_result)

        if valid_records:
            write_results(file_path, (
                updated if updated is not None else original
                for original, updated in zip(records, updated_records)
            ))

            logger.info(f"Updated records saved to {file_path}")

    logger.info(
        f"Program evaluation completed. Total records processed: {total_processed}")

    if correlation_results:
        logger.info("\n" + "="*80)
        logger.info("Correlation between programmatic and judge scores")
        logger.info("="*80)

        correlation_df = pd.DataFrame(correlation_results)
        logger.info("\n" + correlation_df.to_string(index=False))

        correlation_csv = RESULT_DIR / f"correlation_analysis_{MODEL_FILE_NAME}.csv"
        correlation_df.to_csv(correlation_csv, index=False, encoding='utf-8-sig')
        logger.info(f"\nCorrelation analysis saved to: {correlation_csv}")

        logger.info("\nSummary:")
        logger.info(f"  Mean Pearson correlation: {correlation_df['pearson_r'].mean():.4f}")
        logger.info(f"  Mean Spearman correlation: {correlation_df['spearman_r'].mean():.4f}")
        logger.info(f"  Mean MAE: {correlation_df['mae'].mean():.4f}")
        logger.info(f"  Mean RMSE: {correlation_df['rmse'].mean():.4f}")
        logger.info("="*80 + "\n")


def main(args):
    if args.program:
        program_evaluate_all_records(args)
        return
    if args.fix:
        if args.overwrite:
            reassess_all_records_with_overwrite(args)
        else:
            reassess_null_scores(args)
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(REASON_DIR.glob("*_case.jsonl")):
        if "all" in args.task or path.stem.removesuffix("_case") in args.task:
            process_jsonl_file(path, args)
    calculate_average_scores()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UrbanGeoEval reasoning tasks.")
    parser.add_argument(
        "--task", nargs="+", default=["all"],
        help="Task names from the local dataset, or 'all' for a new evaluation.",
    )
    parser.add_argument(
        "--judge_model", default=os.getenv("JUDGE_MODEL_NAME"),
        help="Judge model ID, or set JUDGE_MODEL_NAME.",
    )
    parser.add_argument("--fix", action="store_true", help="Reassess records with missing scores.")
    parser.add_argument("--overwrite", action="store_true", help="With --fix, reassess all records.")
    parser.add_argument(
        "--program", action="store_true",
        help="Recompute answer scores for distance, direction, and planning from saved responses.",
    )
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers.")
    args = parser.parse_args()
    if not MODEL_NAME:
        parser.error("Set MODEL_NAME to the model ID used for generation or saved results.")
    if not args.judge_model:
        parser.error("Set JUDGE_MODEL_NAME or pass --judge_model.")
    main(args)
