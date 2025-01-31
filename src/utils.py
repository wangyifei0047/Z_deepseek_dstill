#!/usr/bin/env python3
import string
from typing import List
import regex
import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy
import os
from tqdm import tqdm
from xopen import xopen
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar
T = TypeVar("T")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(current_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)
logger = logging.getLogger(__name__)

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))




def get_metrics_for_example(example,METRICS):
    # breakpoint()
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    # breakpoint()
    model_answer = model_answer.split("\n</think>\n\n")[-1].strip()
    # model_answer = model_answer.split("")[-1].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
        
    return (example_metrics, example)

def evaluate_qa_data(input_path,output_path=None,sample_num = None):
    METRICS = [
    (best_subspan_em, "best_subspan_em"),]
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)
    if sample_num:
        all_examples = all_examples[:sample_num]
    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        all_example_metrics.append(get_metrics_for_example(example,METRICS))
    # Average metrics across examples
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        print(f"{metric_name}: {average_metric_value}")
        logger.info(f"{metric_name}: {average_metric_value}")

    summary_path = os.path.join(os.path.dirname(input_path),"A_metrics_summary.txt")
    with xopen(summary_path,"a") as f:
        f.write(f"{input_path.split('/')[-1].split('.jsonl.gz')[0]}\n{metric_name}: {average_metric_value}\n\n")
    if output_path:
        with xopen(output_path, "w") as f:
            for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")
# evaluate_qa_data("/root/projects/position_bias/lost_in_the_middle/results/original/Llama-3.1-8B-Instruct_nq-open-50_total_documents_gold_at_0.jsonl.gz")
@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))
