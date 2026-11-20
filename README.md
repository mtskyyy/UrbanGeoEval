# UrbanGeoEval

**A City-Scale Benchmark for Evaluating Large Language Models in Geospatial Reasoning**

**ACL 2026 Main Conference · Long Papers**

[ACL paper](https://aclanthology.org/2026.acl-long.1867/) · [Hugging Face collection](https://huggingface.co/collections/SkyyyyyMT/urbangeoeval) · [Choice dataset](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-choice) · [Reason dataset](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-reason) · [Quickstart](#evaluation-quickstart) · [Citation](#citation)

## Benchmark overview

UrbanGeoEval is a benchmark for evaluating **urban geospatial intelligence in large language models**: how well models know cities, interpret geographic information, and solve spatial problems in realistic urban environments. It spans points of interest, road networks, and urban areas, connecting foundational spatial skills with applications such as itinerary planning, navigation, and location-based recommendation.

The benchmark separates **geographic knowledge** from **geospatial reasoning**, making it possible to diagnose whether a model struggles to recall urban facts or to reason with the information it is given. Its two complementary modules combine broad geographic coverage with fine-grained evaluation of spatial capabilities.

<p align="center">
  <img src="assets/figure-1.png" alt="A restaurant recommendation illustrates how a false place recall can lead to an invalid route." width="460">
</p>

*Figure 1. A geographic knowledge error can lead to a subsequent reasoning error.*

- **Knowledge (Choice): urban knowledge across 18 cities and 6 continents.** OpenStreetMap-derived multiple-choice questions cover six dimensions—nodes, areas, roads, types, functions, and surroundings—through 20 question templates, probing geographic precision, entity attributes, and spatial relationships.
- **Reasoning (Reason): foundational skills through complex spatial decisions.** Seven task families cover distance, direction, spatial relations, itinerary planning, trajectory prediction, navigation, and point-of-interest recommendation. Open-ended questions provide the necessary geographic context, including coordinates and geometry, to assess spatial computation and inference independently of factual recall.
- **Evaluation: reasoning quality and answer correctness.** The paper combines task-specific programmatic checks with rubric-guided LLM judging to assess both the reasoning process and the final answer. This supports analysis of geographic biases, knowledge granularity, and the foundational spatial skills underlying more complex tasks.

<p align="center">
  <img src="assets/figure-2.png" alt="UrbanGeoEval architecture showing separate Knowledge and Reasoning modules and their tasks." width="800">
</p>

*Figure 2. The Knowledge module generates multiple-choice questions, and the Reasoning module supplies context for open-ended spatial problems. Figures are reproduced from [Bao et al. (ACL 2026)](https://aclanthology.org/2026.acl-long.1867/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

## Datasets

| Module | Hugging Face | Release size | Local evaluation inputs |
| --- | --- | ---: | --- |
| **Knowledge (Choice)** | [UrbanGeoEval-choice](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-choice) | 113,851 questions | [`urban_data/geoqa_choice/`](urban_data/geoqa_choice) · 18 cities across 6 continents, 20 question templates |
| **Reasoning (Reason)** | [UrbanGeoEval-reason](https://huggingface.co/datasets/SkyyyyyMT/UrbanGeoEval-reason) | 3,163 questions | [`urban_data/geoqa_reason/`](urban_data/geoqa_reason) · 3,148 questions across 7 task families |

Each Hugging Face release is a single JSONL file. The scripts below read the local per-task files. The paper reports about 54,000 Knowledge questions in Table 1 and 3,148 Reasoning questions in Table 3.

## Leaderboard

Results from the [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1867/), ranked separately for each module. Higher is better for all scores. These results use the paper's evaluation sets and protocol.

### Knowledge (Choice)

Accuracy (%) across six knowledge dimensions, averaged across 18 cities (Table 4).

| Rank | Model | Node | Area | Road | Type | Function | Surroundings | Overall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GPT-4o | 38.83 | 68.18 | 59.59 | 76.15 | 53.29 | 28.89 | **54.59** |
| 2 | Qwen2.5-72B-Instruct | 30.94 | 59.25 | 36.54 | 78.86 | 53.56 | 27.11 | **47.64** |
| 3 | Llama-3.1-70B-Instruct | 30.05 | 56.02 | 38.24 | 77.63 | 49.09 | 22.55 | **45.71** |
| 4 | Qwen2.5-7B-Instruct | 23.65 | 46.97 | 28.57 | 74.28 | 48.60 | 25.46 | **40.57** |
| 5 | Llama-3.1-8B-Instruct | 22.96 | 45.48 | 28.03 | 66.69 | 32.04 | 20.43 | **35.89** |

### Reasoning (Reason)

Average total score on a normalized 0–100 scale, incorporating reasoning-process and final-answer scores across seven task families (Figure 3). Task-level breakdowns are provided in Table 5 of the paper.

| Rank | Model | Overall score |
| ---: | --- | ---: |
| 1 | Qwen3.5-397B-A17B | **85.6** |
| 2 | o3-mini | 85.3 |
| 3 | Qwen3-32B | 82.1 |
| 4 | MiniMax-M2.5 | 81.2 |
| 5 | QwQ-32B | 80.1 |
| 6 | DeepSeek-R1 | 78.2 |
| 7 | DeepSeek-V3 | 75.5 |
| 8 | GPT-4o | 74.3 |
| 9 | Qwen3-8B | 73.6 |
| 10 | Qwen2.5-72B-Instruct | 69.4 |
| 11 | Phi-4 | 69.1 |
| 12 | DeepSeek-R1-32B | 67.4 |
| 13 | Gemma-3-27B | 67.2 |
| 14 | Llama-3.1-70B-Instruct | 65.1 |
| 15 | Qwen2.5-7B-Instruct | 57.6 |
| 16 | Llama-3.1-8B-Instruct | 55.2 |
| 17 | DeepSeek-R1-7B | 49.6 |
| 18 | DeepSeek-R1-1.5B | 37.3 |

## Evaluation quickstart

Run the commands from the repository root in a Python 3.12 environment.

```bash
python -m pip install -r requirements.txt
```

The scripts call an OpenAI-compatible Chat Completions endpoint. The default is `http://localhost:8001/v1`. Set `OPENAI_BASE_URL` for a different endpoint and `OPENAI_API_KEY` if it requires authentication. Use the model IDs advertised by that endpoint.

### Knowledge (Choice)

```bash
python -m evaluate.urban_eval.eval_qa_choice \
  --city_name Berlin --model_name your-model-id
```

Use `--city_name all` for all 18 cities. `--max_valid` determines the sample size for task files with more than twice that many questions. Per-task accuracy, response logs, and checkpoints are saved under `evaluate/urban_eval/results/geoqa_choice/`.

### Reasoning (Reason)

```bash
export MODEL_NAME=your-model-id
export JUDGE_MODEL_NAME=your-judge-model-id
python -m evaluate.urban_eval.eval_qa_reason --task distance
```

The LLM judge scores the reasoning process and final answer separately using the task-specific [rubrics](evaluate/urban_eval/reason_prompt.py). Use `--task all` to evaluate the full local Reasoning set. Results are saved under `evaluate/urban_eval/results/geoqa_reason/<model>/`.

To recompute answer scores for saved distance, direction, and planning responses, run the programmatic scoring pass.

```bash
python -m evaluate.urban_eval.eval_qa_reason --program
```

This pass uses the judge to extract answer values and preserves the existing reasoning score. The released script uses circular bearing error for direction and Levenshtein edit distance for planning. Use `--fix` to retry incomplete judge scores or `--fix --overwrite` to rejudge all saved responses.

## Citation

```bibtex
@inproceedings{bao-etal-2026-urbangeoeval,
  title     = "{UrbanGeoEval}: A City-Scale Benchmark for Evaluating Large Language Models in Geospatial Reasoning",
  author    = "Bao, Mutian and Qi, Qiuyi and Liang, Tian and Zhang, Jinjian and
               Zhou, Wei and Kong, Ming and Mo, Linjian and Zhu, Qiang",
  booktitle = "Proceedings of ACL 2026 (Volume 1: Long Papers)",
  year      = "2026",
  pages     = "40183--40223",
  publisher = "Association for Computational Linguistics",
  doi       = "10.18653/v1/2026.acl-long.1867",
  url       = "https://aclanthology.org/2026.acl-long.1867/"
}
```
