import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", required=True, type=str)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_kw_args(parser):
    args, _ = parser.parse_known_args()
    table = pd.read_csv(args.results_path, index_col=0).reset_index()
    for col in table.columns:
        if table[col].dtype == "bool":
            parser.add_argument(
                f"--{col}",
                type=str2bool,
                nargs="?",
                const=True,
                default=None,
                choices=[True, False],
            )
        else:
            parser.add_argument(f"--{col}", type=type(table[col].iloc[0]), default=None)
    parser.add_argument(
        "--output_value", type=str, required=True, choices=table.columns
    )
    return parser


def filter_table(table_path, model_parameter, **kwargs):
    table = pd.read_csv(table_path)
    for key, value in kwargs.items():
        if value is not None and key in table.columns:
            table = table.loc[table[key] == value]
    if len(table) == 0:
        raise ValueError("No results found")
    elif len(table) > 1:
        raise ValueError(f"Multiple results found for {kwargs} (n={len(table)})")
    return table.iloc[0][model_parameter]


if __name__ == "__main__":
    parser = add_kw_args(parser)

    args = parser.parse_args()
    print(filter_table(args.results_path, args.output_value, **vars(args)))
