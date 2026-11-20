# UrbanGeoEval

UrbanGeoEval is a city-scale benchmark for evaluating urban geographic knowledge and spatial reasoning in language models. See the [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1867/) and the [Hugging Face collection](https://huggingface.co/collections/SkyyyyyMT/urbangeoeval).

## Data

| Module | Local evaluation inputs | Hugging Face release |
| --- | --- | --- |
| Choice | `urban_data/geoqa_choice/<city>/<task>.csv` | [UrbanGeoEval-choice](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-choice), 113,851 multiple-choice questions across 18 cities and 20 task templates |
| Reason | `urban_data/geoqa_reason/<task>_case.jsonl` | [UrbanGeoEval-reason](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-reason), 3,163 open-ended questions |

The evaluation scripts use the local per-task files. The local Reason inputs contain 3,148 questions, matching the counts in the paper. The Hugging Face releases are single JSONL files.

## Evaluation

From the repository root, install the dependencies in a Python 3.12 environment.

```bash
python -m pip install -r requirements.txt
```

Serve the models through an OpenAI-compatible Chat Completions endpoint. The default endpoint is `http://localhost:8001/v1`. Set `OPENAI_BASE_URL` to use another endpoint, and set `OPENAI_API_KEY` if it requires authentication. Use the exact model IDs advertised by your endpoint.

### Choice

```bash
python -m evaluate.urban_eval.eval_qa_choice \
  --city_name Berlin --model_name your-model-id
```

Use `--city_name all` to include every city. The evaluator writes accuracy summaries, per-question logs, and checkpoints under `evaluate/urban_eval/results/geoqa_choice/`. The `--max_valid` setting samples that many rows from each task file that contains more than twice that number of rows.

### Reason

```bash
export MODEL_NAME=your-model-id
export JUDGE_MODEL_NAME=your-judge-model-id
python -m evaluate.urban_eval.eval_qa_reason --task distance
```

The judge scores the reasoning process and final answer using the rubrics in [`reason_prompt.py`](evaluate/urban_eval/reason_prompt.py). Use `--task all` for the full local Reason set. Per-question results are written under `evaluate/urban_eval/results/geoqa_reason/<model>/`.

To recompute answer scores on saved distance, direction, and planning responses, run:

```bash
python -m evaluate.urban_eval.eval_qa_reason --program
```

This pass uses the judge model to extract answer values, then applies programmatic scoring. The reference implementation uses circular bearing error for direction and Levenshtein edit distance for planning. Use `--fix` to retry records with incomplete judge scores or `--fix --overwrite` to rejudge all saved responses.
