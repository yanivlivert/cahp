import numpy as np
from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    from sklearn.metrics import f1_score
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def glue_compute_metrics(task_name, preds, labels):
    if task_name == "cola":
        from sklearn.metrics import matthews_corrcoef
        return {"mcc": matthews_corrcoef(labels, preds)}

    elif task_name == "sst-2" or task_name == "qnli" or task_name == "rte" or task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}

    elif task_name == "mrpc" or task_name == "qqp":
        return acc_and_f1(preds, labels)

    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)

    elif task_name == "mnli" or task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}

    else:
        raise KeyError(task_name)
