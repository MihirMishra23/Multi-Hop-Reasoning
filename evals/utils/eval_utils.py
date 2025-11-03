"""Utilities for computing evaluation metrics given prediction files
"""
import sys
import ujson as json
import re
import string
from collections import Counter
import logging
import traceback
from glob import glob

import numpy as np
import pandas as pd
from joblib import Memory, expires_after

memory = Memory("cachedir")

from . import constants
from .utils import load_data

# hard-code the order of the models for the plot
# otherwise, the order will be alphabetical (and the model size will not be in order)
MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct-turbo",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "microsoft/phi-3.5-moe-instruct",
    "microsoft/phi-3.5-mini-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
    "qwen/qwen2.5-32b-instruct",
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwen2.5-1.5b-instruct",
]


def pivot_mean_std(acc_mean_std, metric, independent_variable="_split", enforce_order=True):
    """Pivot acc_mean_std so that the specified independent variable becomes the rows

    Args:
        acc_mean_std (pd.DataFrame): The dataframe to pivot
        metric (str): The metric to pivot on
        independent_variable (str): The independent variable to pivot on
        enforce_order (bool): Whether to enforce the order of the models
    Returns:
        df_mean (pd.DataFrame): The mean of the metric
        df_std (pd.DataFrame): The standard deviation of the metric
    """
    assert (metric, "mean") in acc_mean_std.columns
    assert (metric, "std") in acc_mean_std.columns
    assert ("_model", "") in acc_mean_std.columns
    assert (independent_variable, "") in acc_mean_std.columns

    df_mean = acc_mean_std.pivot(index="_model", columns=independent_variable, values=(metric, "mean"))
    # change the column names to integers
    df_mean.columns = df_mean.columns.astype(int)
    # reorder the columns in ascending order
    df_mean = df_mean[sorted(df_mean.columns)]
    if enforce_order:
        row_order = [name for name in MODELS if name in df_mean.index]
        df_mean = df_mean.loc[row_order]

    df_std = acc_mean_std.pivot(index="_model", columns=independent_variable, values=(metric, "std"))
    # change the column names to integers
    df_std.columns = df_std.columns.astype(int)
    df_std = df_std[sorted(df_std.columns)]
    if enforce_order:
        row_order = [name for name in MODELS if name in df_std.index]
        df_std = df_std.loc[row_order]
    return df_mean, df_std


################ Utils for getting evaluation data ################
@memory.cache(cache_validation_callback=expires_after(hours=4))
def _get_preds(output_dir, method):
    """Get predictions from the output directory corresponding to method `method`

    Args:
        output_dir (str): path to the output directory
        method (str): method used for inference (e.g., zeroshot, fewshot, etc.)

    Returns:
        pd.DataFrame: a dataframe containing the predictions
    """
    # get all files in the output directory
    # NOTE: the actual filenames do not matter, since each row also contains
    # the model, split, batch_size, batch_number, and seed in the metadata and sampling params fields
    files = glob(f"{output_dir}/preds/{method}/*.json")
    # sort the files by the batch number
    files = sorted(files, key=lambda x: int(re.search(r"bn=(\d+)", x).group(1)))

    if len(files) == 0:
        logging.warning(f"No files found in {output_dir}/preds/{method}/*.json")
        return pd.DataFrame()

    df_list = []
    # keys to create auxiliary columns that are useful for analysis
    METADATA = ["model", "split", "batch_size", "batch_number", "type"]
    SAMPLING_PARAMS = ["seed"]
    for filename in files:
        logging.info(f"Reading from {filename}...")
        df = pd.read_json(filename, orient="index", dtype=False)
        # add new columns corresponding to the metadata
        for key in METADATA:
            df["_" + key] = df["metadata"].apply(lambda x: x[key])
        # add new columns corresponding to the sampling parameters
        for key in SAMPLING_PARAMS:
            df["_" + key] = df["inference_params"].apply(lambda x: x[key])
        # drop the metadata column
        df = df.drop(columns=["metadata"])
        df_list.append(df)
    # concatenate all dataframes
    df_preds = pd.concat(df_list)
    # and add a new index from 0 to len(df_preds) so that we can save the dataframe to a json file
    # assign the old index to a new column called 'id'
    df_preds = df_preds.reset_index(names="id")
    return df_preds


@memory.cache(cache_validation_callback=expires_after(hours=4))
def _get_qa_pairs(dataset: str, splits: list[str], from_local: bool = False):
    df_list = []
    for split in splits:
        # NOTE: using Dataset.to_pandas() casts lists to numpy arrays
        # which is not JSON serializable. Thus, we use pd.DataFrame() instead
        df = pd.DataFrame(load_data(dataset, split, from_local)["qa_pairs"])
        # # set index to id
        # df = df.set_index('id')
        # convert template column to string
        df["template"] = df["template"].apply(lambda x: " ".join(x))
        # compute the number of hops by taking the length of the prolog query
        df["hops"] = df["prolog"].apply(lambda x: len(x["query"]))
        # determine whether a question is an aggregation question or not
        df["aggregation"] = df["prolog"].apply(lambda x: "aggregate_all" in " ".join(x["query"])).astype(int)
        # determine the number of solutions to each question
        df["solutions"] = df["answer"].apply(lambda x: len(x))
        df_list.append(df)
    # merge on the index
    df_qa_pairs = pd.concat(df_list)
    return df_qa_pairs


@memory.cache(cache_validation_callback=expires_after(hours=4))
def get_predictions_with_qa(
    output_dir: str,
    method: str,
    dataset: str,
    from_local: bool = False,
):
    """Get the predictions with the qa pairs

    First reads the predictions from the output directory, then joins with the qa pairs.
    NOTE: The results are cached for 4 hours at `cachedir` (see `memory` object).
    To invalidate the cache, delete the `cachedir` folder.
    NOTE: to include the scores, use `get_evaluation_data`.

    Args:
        output_dir (str): path to the output directory
        method (str): method used for inference (e.g., zeroshot, fewshot, etc.)
        dataset (str): dataset name (e.g., "mlcore/phantom-wiki", "mlcore/phantom-wiki-v0.2")
        from_local (bool) : if loading the data from a local folder.
            Default is False.

    Returns:
        pd.DataFrame: a dataframe containing the evaluation data,
            including the predictions, the qa pairs (with auxiliary columns),
            and per-instance evaluation metrics
    """
    # get the predictions
    df_preds = _get_preds(output_dir, method)
    if df_preds.empty:
        return df_preds
    # get unique splits
    splits = df_preds["_split"].unique().tolist()
    # get the qa pairs
    df_qa_pairs = _get_qa_pairs(dataset, splits, from_local)
    # join with original qa pairs to get additional information about
    # the prolog queries and the templates
    df = df_preds.merge(df_qa_pairs, on="id", how="left")

    # add a column for data generation parameters
    df["_depth"] = (
        df["_split"].apply(lambda x: re.match(r"depth_(\d+)_size_(\d+)_seed_(\d+)", x).group(1)).astype(int)
    )
    df["_size"] = (
        df["_split"].apply(lambda x: re.match(r"depth_(\d+)_size_(\d+)_seed_(\d+)", x).group(2)).astype(int)
    )
    df["_data_seed"] = (
        df["_split"].apply(lambda x: re.match(r"depth_(\d+)_size_(\d+)_seed_(\d+)", x).group(3)).astype(int)
    )
    return df


@memory.cache(cache_validation_callback=expires_after(hours=4))
def get_evaluation_data(
    output_dir: str,
    method: str,
    dataset: str,
    from_local: bool = False,
    sep: str = constants.answer_sep,
):
    """Get the predictions with scores

    Args:
        output_dir (str): path to the output directory
        method (str): method used for inference (e.g., zeroshot, fewshot, etc.)
        dataset (str): dataset name (e.g., "mlcore/phantom-wiki", "mlcore/phantom-wiki-v0.2")
        from_local (bool) : if loading the data from a local folder.
            Default is False.
        sep (str): separator when pre-processing pred/true strings.
            Default is `constants.answer_sep`.

    Returns:
        pd.DataFrame: a dataframe containing the predictions with scores
    """
    df = get_predictions_with_qa(output_dir, method, dataset, from_local)
    # join the true answers with the appropriate separator since the scoring functions expect strings
    try:
        df["EM"] = df.apply(lambda x: exact_match(x["pred"], sep.join(x["true"]), sep=sep), axis=1)
        df["precision"] = df.apply(lambda x: precision(x["pred"], sep.join(x["true"]), sep=sep), axis=1)
        df["recall"] = df.apply(lambda x: recall(x["pred"], sep.join(x["true"]), sep=sep), axis=1)
        df["f1"] = df.apply(lambda x: f1(x["pred"], sep.join(x["true"]), sep=sep), axis=1)
        return df
    except ValueError:
        logging.warning(f"Error in computing scores for {traceback.format_exc()}, returning empty dataframe.")
        return pd.DataFrame()


def mean(x):
    """Aggregation function that computes the mean of a given metric"""
    return x.mean()


def std(x):
    """Aggregation function that computes the standard error of the mean of a given metric"""
    return x.std(ddof=1) / np.sqrt(len(x))

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])

