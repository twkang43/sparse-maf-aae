import os
import numpy as np
from sklearn import metrics
from data_preprocessing.voraus_ad import ANOMALY_CATEGORIES

def get_metrics(results):
    aurocs = []

    # Calculate AUROC & AUPR per anomaly category.
    for category in ANOMALY_CATEGORIES:
        dfn = results[(results["category"] == category.name) | (~results["anomaly"])]

        fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
        auroc = metrics.auc(fpr, tpr)*100
        aurocs.append(auroc)

    # Calculate the AUROC, AUPR mean over all categories.
    print(f"auroc : {np.mean(aurocs)}")

    return aurocs

def calculate_elapsed_time(elapsed_time):
    Q1 = np.percentile(elapsed_time, 25)
    Q3 = np.percentile(elapsed_time, 75)
    return [time for time in elapsed_time if Q1 <= time <= Q3]