import numpy as np
import torch
import scipy
import os
import pandas as pd

from config import parse_args
from pfn import PFN
from methods.ftknn import train_ft_knn, eval_ft_knn
from methods.pfknn import eval_pfknn
from methods.vanilla import eval_tabpfn

from dataset import PFNDataset
from utils import setup_experiment, compute_metrics, save_numpy, create_dataloaders

if __name__ == "__main__":
    args = parse_args()

    pfnDataset = PFNDataset(args)
    datasets = pfnDataset.load()
    writer, results_file, experiment_path = setup_experiment(args)

    model, _ = PFN.load_old(device=args.device, path=args.model_path)
    model.eval()

    num_datasets = 0
    for data in datasets:
        if not data:  # in case data contains None
            print("data contains None")
            continue

        dataset_name = data["dataset_info"]["name"]
        print(dataset_name)

        num_datasets += 1

        create_dataloaders(args, data)

        import time
        t0 = time.time()

        # Train and Eval
        if args.method == "vanilla": # sub-sampling
            loss, pred = eval_tabpfn(args, model, data, data["test_loader"])
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "knn": # TabPFN-kNN
            if args.dynamic:
                args.context_length = min(
                    int(10 * np.sqrt(len(data["X_train"]))), args.context_length
                )
            loss, pred = eval_pfknn(args, model, data, data["test_loader"])
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "ft": # LoCalPFN
            model, _ = PFN.load_old(device=args.device, path=args.model_path)
            model.train()

            train_ft_knn(args, model, data, writer, experiment_path)
            model.load_state_dict(
                torch.load(f"{experiment_path}/data/{dataset_name}/model_best.pth")
            )
            model.eval()
            loss, pred = eval_ft_knn(args, model, data["test_loader"], data)

            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        else:
            raise ValueError("Invalid Method")

        total_time = time.time() - t0

        # Logging
        if args.method in ["vanilla", "knn", "dist", "ft", "dt"]:
            pred = scipy.special.softmax(pred, axis=1)

        acc, f1, auc = compute_metrics(data["y_test"], pred)
        if args.timing:
            print(
                "Best: Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}, AUC: {:.4f}, Time: {:6f}".format(
                    loss, acc, f1, auc, total_time
                )
            )
            results_file.write(f"{dataset_name},{acc},{f1},{auc},{total_time}\n")
            results_file.flush()
        else:
            print(
                "Best: Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(
                    loss, acc, f1, auc
                )
            )
            results_file.write(f"{dataset_name},{acc},{f1},{auc}\n")
            results_file.flush()

    if not args.timing:
        df = pd.read_csv(
            os.path.join(experiment_path, "results.csv"),
            sep=",",
            header=None,
            names=["name", "acc", "f1", "auc"],
        )
        medians = df.iloc[-num_datasets:, 1:].astype(float).median()
        print(medians)
        medians.to_csv(results_file, mode="a", header=True, index=False)

