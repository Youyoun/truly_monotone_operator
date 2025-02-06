import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from toolbox.metrics import MetricsDictionary

HTML_MODEL = (
    "/home/youyoun/Projects/monotone_operator/SimpleMonotoneOperators/model.html"
)


def plot_to_base64(figure: plt.Figure) -> str:
    my_stringIObytes = io.BytesIO()
    figure.savefig(my_stringIObytes, format="jpg")
    my_stringIObytes.seek(0)
    return base64.b64encode(my_stringIObytes.read()).decode()


def plot_to_html(figure: plt.Figure) -> str:
    base64_fig = plot_to_base64(figure)
    return f"<img src='data:image/png;base64, {base64_fig}'>"


def get_template_html(
    run_name: str = None,
    args: Dict[str, Any] = None,
    logfile: Path = None,
    performance: Dict[str, Any] = None,
    train_results: MetricsDictionary = None,
    images_dir: Path = None,
    ev_results: Tuple[List[float], List[float]] = None,
    test_results_paths: List[str] = None,
    model_paths: Dict[str, str] = None,
):
    template = open(HTML_MODEL).read()
    template = template.format(
        title="results.html",
        run_name=run_name,
        logfile=f"file://{logfile.absolute()}",
        args=get_args_html(args),
        performance=get_performance_html(performance),
        trainResults=get_train_results_html(train_results, images_dir),
        testResults=get_test_results_html(test_results_paths, ev_results),
        models=get_models_html(model_paths),
    )
    return template


def get_args_html(args: Dict[str, Any]) -> str:
    newline = "\n"
    return f"""
    <ul>
        {newline.join(f'<li>{name}: {value}</li>' for name, value in args.items())}
    </ul>
    """


def get_performance_html(performance: Dict[str, Any]) -> str:
    return pd.DataFrame([performance]).to_html(index=False)


def get_train_results_html(metrics: MetricsDictionary, train_images_dir: Path) -> str:
    ims = []
    for metric, values in metrics.get_all().items():
        fig, ax = plt.subplots()
        ax.plot(values)
        ax.set_title(f"Metrics: {metric}")
        fig_html = plot_to_html(fig)
        ims.append(fig_html)
    train_ims = "".join(
        f'<div>{path.name}<img src="{train_images_dir.name}/{path.name}"></div>'
        for path in sorted(train_images_dir.iterdir())
    )
    return "<div>" + "".join(ims) + "</div>" + "\n" + train_ims


def get_test_results_html(
    test_results: List[str], evs: Tuple[List[float], List[float]]
) -> str:
    fig, ax = plt.subplots(1, 2)
    ax[0].hist([evs[0]], bins=50)
    ax[0].set_title(r"$\lambda_{min}$")
    ax[1].hist([evs[1]], bins=50)
    ax[1].set_title(r"$\lambda_{max}$")
    fig_html = plot_to_html(fig)
    return (
        fig_html
        + "\n"
        + "".join(f'<img src="{path}">' for path in sorted(test_results))
    )


def get_models_html(model_paths: Dict[str, str]) -> str:
    newline = "\n"
    return f"""
    <div class="row">
        {newline.join(f'<div class="col-sm"><h3>{name}</h3><img src="{path}"></div>' for name, path in model_paths.items())}
    </div>
    """
