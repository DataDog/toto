"""
This script evaluates a TOTO model on LSF datasets using checkpoints obtained from MLFlow runs.
It fetches the best checkpoints based on validation metrics, evaluates each checkpoint on LSF datasets,
and logs the evaluation results as CSV artifacts in MLFlow.

Example usage:

ray job submit \
    --address [RAY_CLUSTER_ADDRESS] \
    --runtime-env ./lsf_eval_runtime.yaml -- \
    python scripts/run_lsf_eval.py \
    --mlflow-run-id [MLFLOW_RUN_ID]
"""

# TODO(Anna) - update the description of the script and its usage

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import ray
import torch
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from toto.evaluation.lsf.lsf_datasets import LSFDatasetName
from toto.evaluation.lsf.lsf_evaluator import LSFEvaluator
from toto.inference.gluonts_predictor import Multivariate
from toto.model.toto import Toto

LSF_DATASETS_LOCAL_PATH = "./data/lsf_datasets"
CPUS_PER_WORKER = 4


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a TOTO model on LSF datasets.")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for evaluation. Will be used to parallelize across datasets.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            # "electricity",
            "weather",
        ],
        help="List of LSF datasets to evaluate on.",
    )

    parser.add_argument(
        "--prediction-lengths",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="List of prediction lengths to evaluate on.",
    )

    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[2048],
        help="List of context lengths to evaluate on.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples to draw from the model.",
    )

    parser.add_argument(
        "--data-split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Data split to evaluate on.",
    )

    parser.add_argument(
        "--eval-stride",
        type=int,
        default=512,
        help="Stride to use for evaluation.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for evaluation. Multiply by samples_per_batch to get effective batch size.",
    )  # TODO(Anna) - should we remove this to avoid confusion with samples_per_batch?

    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=256,
        help="Number of samples to draw per batch.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=59,
        help="Seed for reproducibility.",
    )

    parser.add_argument(
        "--use-kv-cache",
        type=bool,
        default=True,
        help="Whether to use key-value caching during inference.",
    )  # TODO(Anna) - should this be configurable?
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="Datadog/Toto-Open-Base-1.0",
        help="Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`."
        "Or a path to a `directory` containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.",
    )

    return parser


@dataclass(frozen=True)
class EvalTask:
    dataset: LSFDatasetName
    checkpoint_path: str
    data_split: str
    prediction_length: int
    context_length: int
    eval_stride: int
    batch_size: int
    num_samples: int
    samples_per_batch: int
    seed: int
    use_kv_cache: bool


def evaluate_checkpoint(task: EvalTask) -> pd.DataFrame:
    """
    Evaluate a TOTO model on LSF datasets.

    Fetches the model from Huggingface Hub, evaluates it on LSF datasets, and returns the evaluation results
    as a DataFrame.
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(task.seed)
    np.random.seed(task.seed)

    model = Toto.from_pretrained(task.checkpoint_path)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.compile(model, mode="max-autotune")
    model.eval()

    evaluator = LSFEvaluator(
        datasets=[task.dataset],
        prediction_lengths=[task.prediction_length],
        context_lengths=[task.context_length],
        num_samples=task.num_samples,
        lsf_path=str(LSF_DATASETS_LOCAL_PATH),
        data_split=task.data_split,
        mode=Multivariate(task.batch_size),
        eval_stride=task.eval_stride,
        samples_per_batch=task.samples_per_batch,
        use_kv_cache=task.use_kv_cache,
    )

    evalutions, _, _ = evaluator.eval(model, task.checkpoint_path)
    return evalutions


@ray.remote(num_cpus=CPUS_PER_WORKER, num_gpus=1)
def evaluate_checkpoints(tasks: list[EvalTask]) -> pd.DataFrame:
    return pd.concat([evaluate_checkpoint(task) for task in tasks])


def main():
    parser = get_parser()
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path

    print(f"Evaluating checkpoint: {checkpoint_path}")

    # Create an evaluation task for each checkpoint
    tasks = [
        EvalTask(
            dataset=LSFDatasetName(dataset),
            checkpoint_path=checkpoint_path,
            data_split=args.data_split,
            prediction_length=prediction_length,
            context_length=context_length,
            eval_stride=args.eval_stride,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            samples_per_batch=args.samples_per_batch,
            seed=args.seed,
            use_kv_cache=args.use_kv_cache,
        )
        for dataset in args.datasets
        for prediction_length in args.prediction_lengths
        for context_length in args.context_lengths
    ]

    # Batch tasks to send to workers
    num_gpus = min(args.num_gpus, len(tasks))
    print(f"Using {num_gpus} GPUs for evaluation.")

    batches: List[List[EvalTask]] = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        batches[i % num_gpus].append(task)

    print(f"Eval tasks per GPU: {[len(batch) for batch in batches]}")

    assert len(batches) == num_gpus, "Each GPU should have a batch of tasks."
    assert all(len(batch) > 0 for batch in batches), "Each batch should have at least one task."
    assert sum(len(batch) for batch in batches) == len(tasks), "All tasks should be assigned to a batch."

    # Distribute tasks to workers to evaluate checkpoints in parallel
    result_refs = [evaluate_checkpoints.remote(batch) for batch in batches]
    task_results = ray.get(result_refs)

    # Combine results and summarize
    results = pd.concat(task_results)
    summary_results = results.groupby(["checkpoint", "dataset"]).mean()
    print(tabulate(results.reset_index(), headers="keys", tablefmt="psql"))  # Table-like format
    print(tabulate(summary_results.reset_index(), headers="keys", tablefmt="psql"))  # Table-like format


if __name__ == "__main__":
    main()
