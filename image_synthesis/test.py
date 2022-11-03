import argparse
import os

import torch
import wandb
from torch.utils.data import DataLoader

from test_dataset import TestDataset
from utils import print_model_metrics_ODF, print_model_metrics_OIF


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root', type=str, default='./output/dataset_128_outlines_r', help='where to get the images from')
    parser.add_argument('--project_name', default=None, help='Wandb project name')
    parser.add_argument('--run_name', default=None, help='Wandb run name')
    parser.add_argument('--n_images', type=int, default=16, help='Number of images to test')
    parser.add_argument('--domain_transfer', '-d', type=bool, default=False, help='domain transfer from fake to real')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    test_set = TestDataset(args.root, do_domain_transfer=args.domain_transfer)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.n_images)

    for batch in test_loader:
        m1_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=1)
        m3_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=3)
        m6_ODF = print_model_metrics_ODF(batch['truemask'], batch['predmask'], matching_distance=6)

        m1_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=1)
        m3_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=3)
        m6_OIF = print_model_metrics_OIF(batch['truemask'], batch['predmask'], matching_distance=6)
        break

    wandb.init(project=args.project_name, name=args.run_name, entity="michelleappel", id=args.run_name, resume=True)

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
