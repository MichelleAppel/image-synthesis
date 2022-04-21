import os

import argparse

import torch
from torch.utils.data import DataLoader

import wandb

from utils import print_model_metrics_ODF, print_model_metrics_OIF
from test_dataset import TestDataset

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root', type=str, default='output', help='where to get the images from')
    parser.add_argument('--project_name', default=None, help='Wandb project name')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    test_set = TestDataset(args.root)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=64)

    for batch in test_loader:
        m1_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=1)
        m3_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=3)
        m6_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=6)

        m1_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=1)
        m3_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=3)
        m6_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=6)
        break

    wandb.init(project=args.project_name, entity="michelleappel")

    i = 1
    data = [[x, y] for (x, y) in zip([1,3,6], [m1_ODF[i], m3_ODF[i], m6_ODF[i]])]
    table = wandb.Table(data=data, columns = ["matching distance", "AP"])
    wandb.log({"AP_ODF" : wandb.plot.line(table, "matching distance", "AP",
            title="ODF AP")})

    data = [[x, y] for (x, y) in zip([1,3,6], [m1_OIF[i], m3_OIF[i], m6_OIF[i]])]
    table = wandb.Table(data=data, columns = ["matching distance", "AP"])
    wandb.log({"AP_OIF" : wandb.plot.line(table, "matching distance", "AP",
            title="OIF AP")})

    i = 0
    data = [[x, y] for (x, y) in zip([1,3,6], [m1_ODF[i], m3_ODF[i], m6_ODF[i]])]
    table = wandb.Table(data=data, columns = ["matching distance", "F1"])
    wandb.log({"F1_ODF" : wandb.plot.line(table, "matching distance", "F1",
            title="ODF F1")})

    data = [[x, y] for (x, y) in zip([1,3,6], [m1_OIF[i], m3_OIF[i], m6_OIF[i]])]
    table = wandb.Table(data=data, columns = ["matching distance", "F1"])
    wandb.log({"F1_OIF" : wandb.plot.line(table, "matching distance", "F1",
            title="OIF F1")})
