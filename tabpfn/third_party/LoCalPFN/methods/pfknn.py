from tqdm import tqdm
import numpy as np
import torch

from utils import MulticlassFaiss, get_sizes_per_class, pad_x, SingleclassFaiss

@torch.no_grad()
def eval_pfknn(args, model, data, data_loader, normalize_data=True):
    X_train = data["X_train_one_hot"] if args.use_one_hot_emb else data["X_train"]
    y_train = data["y_train"]
    X_train_one_hot = data["X_train_one_hot"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    faiss_knn = SingleclassFaiss(
        X_train_one_hot if args.onehot_retrieval else X_train,
        y_train,
    )
    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # eval
    y_pred = []
    loss = 0

    for X, y, X_oh in tqdm(data_loader):
        cur_batch_size, cur_num_features = X.shape[0], X.shape[1]

        indices_X_nni, y_nni = faiss_knn.get_knn_indices(
            X_oh if args.onehot_retrieval else X.numpy(),
            sum(sizes_per_class),
        )
        X_nni = X_train[indices_X_nni]
        X_nni = np.swapaxes(X_nni, 0, 1)
        y_nni = np.swapaxes(y_nni, 0, 1)

        X_nni, y_nni = (
            pad_x(torch.Tensor(X_nni)).to(args.device),
            torch.Tensor(y_nni).to(args.device),
        )
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        missing_indices = np.where(indices_X_nni[0] == -1)[0]
        if len(missing_indices) > 0:
            miss_idx = missing_indices[0]
            X_nni = X_nni[:miss_idx]
            y_nni = y_nni[:miss_idx]

        logits = model(
            x_src=torch.cat([X_nni, X], dim=0),
            y_src=torch.cat([y_nni, y], dim=0),
            eval_pos=len(X_nni),
            normalization=normalize_data,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=cur_num_features,
        )

        logits = logits.squeeze(0)[:, :num_classes]
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred
