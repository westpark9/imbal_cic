import argparse
import getpass


def parse_args():
    parser = argparse.ArgumentParser(description="PFN Inference")

    # general
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt",
    )
    parser.add_argument("--inf_temperature", type=float, default=0.8)

    # toy dataset
    parser.add_argument("--datasets", choices=["toy", "tabzilla"], default="tabzilla")
    parser.add_argument(
        "--toy_dataset_name", choices=["moons", "blobs"], default="moons"
    )
    parser.add_argument("--n_samples", type=int, default=1000)

    # tabzilla
    parser.add_argument(
        "--datasets_directory",
        type=str,
        default=f"/home/{getpass.getuser()}/tabzilla/TabZilla/datasets/",
    )
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--disable_normalize_data", action="store_true")
    parser.add_argument("--use_nan_datasets", action="store_true")
    parser.add_argument("--clipping_val", type=int, default=10)
    parser.add_argument("--numerical_only", action="store_true")
    parser.add_argument(
        "--filter_size",
        choices=["small", "small_and_medium", "medium", "medium_and_large", "all"],
        default="medium",
    )
    parser.add_argument("--filter_feature", type=int, default=100)
    parser.add_argument("--filter_class", type=int, default=10)
    parser.add_argument("--add_noise_std", type=float, default=0)

    parser.add_argument("--only_dataset", type=str, default="")
    parser.add_argument("--dataset_list_path", type=str, default="")
    parser.add_argument("--timing", action="store_true")

    # methods
    sub_parser = parser.add_subparsers(dest="method", required=False)

    # knn
    knn_parser = sub_parser.add_parser("knn")
    knn_parser.add_argument("--dynamic", action="store_true")
    knn_parser.add_argument("--context_length", type=int, default=1000)
    knn_parser.add_argument("--use_one_hot_emb", action="store_true")
    knn_parser.add_argument("--onehot_retrieval", action="store_true")
    knn_parser.add_argument(
        "--class_choice", choices=["equal", "balance"], default="equal"
    )
    knn_parser.add_argument("--batch_size", type=int, default=512)
    knn_parser.add_argument("--batch_size_inf", type=int, default=512)
    knn_parser.add_argument(
        "--ensemble_dist_folder",
        type=str,
        default="results/tabzilla/default_distill_pfn0/data/",
    )
    knn_parser.add_argument(
        "--embedding", choices=["raw"], default="raw"
    )

    # vanilla
    vanilla_parser = sub_parser.add_parser("vanilla")
    vanilla_parser.add_argument("--context_length", type=int, default=1000)
    vanilla_parser.add_argument("--use_one_hot_emb", action="store_true")
    vanilla_parser.add_argument(
        "--class_choice", choices=["equal", "balance"], default="equal"
    )
    vanilla_parser.add_argument("--batch_size", type=int, default=512)
    vanilla_parser.add_argument("--ensemble_dist", action="store_true")
    vanilla_parser.add_argument(
        "--ensemble_dist_folder",
        type=str,
        default="results/tabzilla/default_distill_pfn0/data/",
    )
    vanilla_parser.add_argument("--integrated", action="store_true")
    vanilla_parser.add_argument("--ensemble", action="store_true")

    # LoCalPFN
    ft_parser = sub_parser.add_parser("ft")
    ft_parser.add_argument("--lr", type=float, default=1e-5)
    ft_parser.add_argument("--opt_weight_decay", type=float, default=0.01)
    ft_parser.add_argument("--context_length", type=int, default=1000)
    ft_parser.add_argument("--train_query_length", type=int, default=1000)

    ft_parser.add_argument(
        "--class_choice", choices=["equal", "balance"], default="equal"
    )
    ft_parser.add_argument("--batch_size", type=int, default=2)
    ft_parser.add_argument("--batch_size_inf", type=int, default=512)

    ft_parser.add_argument("--num_epochs", type=int, default=21)
    ft_parser.add_argument(
        "--early_stopping_metric",
        choices=["negloss", "acc", "f1", "auc"],
        default="auc",
    )
    ft_parser.add_argument("--early_stopping_rounds", type=int, default=100)
    ft_parser.add_argument("--eval_interval", type=int, default=1)
    ft_parser.add_argument("--embedding", type=str, default="raw")
    ft_parser.add_argument("--use_one_hot_emb", action="store_true")

    ft_parser.add_argument("--scheduler", action="store_true")
    ft_parser.add_argument("--num_steps", type=int, default=30)
    ft_parser.add_argument("--better_selection", action="store_true")

    ft_parser.add_argument("--onehot_retrieval", action="store_true")
    ft_parser.add_argument("--save_data", action="store_true")
    ft_parser.add_argument("--exact_knn", action="store_true")
    

    ft_parser.add_argument(
        "--splits_evaluated", choices=["valid", "valid_test", "all"], default="valid"
    )

    return parser.parse_args()
