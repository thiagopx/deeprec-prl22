import json
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager._rebuild()
# from matplotlib.text import Text
from matplotlib import rc

# from utils import load_rgb
from notebooks.utils import replace_underscore

# sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")
rc("font", **{"family": "serif", "serif": ["Nimbus Roman No9 L"]})

import seaborn as sns

sns.set(context="paper", style="whitegrid")

map_dataset = {
    "S-MARQUES": "\\textsc{S-Marques}",
    "S-ISRI-OCR": "\\textsc{S-Isri-Ocr}",
    "S-CDIP": "\\textsc{S-Cdip}",
}
map_query = {
    "opt-l": "\\textsc{Opt-R}",
    "opt-rl": "\\textsc{Opt-RL}",
    "unc-l": "\\textsc{Unc-R}",
    "unc-rl": "\\textsc{Unc-RL}",
    "random": "baseline",
}
map_wload = {"0.1": "0.10", "0.15": "0.15", "0.2": "0.20", "0.25": "0.25"}


def round_close(val):
    rval = round(val + 0.5, 1)
    frac = rval - int(rval)
    new_frac = 0.5 if frac >= 0.5 else 0.0
    return int(rval) + new_frac


def get_dataframe(fname):
    records_ = json.load(open("{}".format(fname)))["records"]
    records = []
    for record_ in records_:
        records.append(record_)
        record = record_.copy()
        # init sol: workload=0 num_iter=1 (chart 1)
        if (record["num_iter"] == 1) and (record["workload"] == 0.1):
            record["workload"] = 0
            record["accuracy"] = record_["accuracy_original"]
            records.append(record)

            # init sol: workload={...} num_iter=0 (chart 2)
            for wload in [0.1, 0.15, 0.2, 0.25]:
                record = record_.copy()
                record["workload"] = wload
                record["curr_iter"] = 0
                record["num_iter"] = 0
                record["accuracy"] = record_["accuracy_original"]
                records.append(record)

    df = pd.DataFrame.from_records(records)

    replace_underscore(df)
    df = df[
        [
            "dataset-test",
            "seed",
            "approach",
            "query-st",
            "solver",
            "workload",
            "accuracy",
            "accuracy-original",
            "curr-iter",
            "num-iter",
            "num-pred-mistakes",
        ]
    ]
    # df["workload"] = (100 * df["workload"]).astype(np.int32)
    df["accuracy"] = 100 * df["accuracy"]
    df["accuracy-original"] = 100 * df["accuracy-original"]
    # df["approach"] = df["approach"].map({"cl": "\\textsc{Deeprec-CL}", "dml": "\\textsc{Deeprec-DML}"})
    return df


def plot_1_iter(df):
    df = df[df["num-iter"] == 1]
    approaches = ["dml", "cl"]

    for approach in approaches:
        df_app = df[df["approach"] == approach]
        fp = sns.FacetGrid(
            hue="query-st",
            hue_order=["opt-l", "opt-rl", "unc-l", "unc-rl", "random"],
            height=3,
            aspect=1.3,
            col="dataset-test",
            col_order=["S-MARQUES", "S-ISRI-OCR", "S-CDIP"],
            sharey=False,
            data=df_app,
            legend_out=True,
        )
        fp = fp.map(
            sns.lineplot,
            "workload",
            "accuracy",
            ci=None,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgewidth=0.1,
        )
        fp = fp.add_legend(title="query st.", fontsize=22, labelspacing=0.25)
        fp.set_axis_labels("$\\alpha_{load}$", "accuracy (\%)", fontsize=26)

        # adjust each axis
        for ax in fp.axes.flatten():
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            dataset = ax.title.get_text()
            dataset = dataset.replace("dataset-test = ", "")
            dataset = map_dataset[dataset]
            ax.set_title(dataset, fontsize=24)
            ax.set_xticks([0, 0.1, 0.15, 0.2, 0.25])
            ax.set_xticklabels(["0", "0.1", "0.15", "0.2", "0.25"])
            for text in ax.get_xticklabels():
                text.set_fontsize(20)
                text.set_usetex(False)
            for text in ax.get_yticklabels():
                text.set_fontsize(20)
                text.set_usetex(False)

        leg = fp._legend
        leg.get_title().set_fontsize(20)
        for text in leg.get_texts():
            text.set_fontsize(20)
            text.set_text(map_query[text.get_text()])

        bb = leg.get_bbox_to_anchor().transformed(
            fp.axes.flatten()[-1].transAxes.inverted()
        )
        dx = 0.1
        dy = 0
        bb.y0 += dy
        bb.y1 += dy
        bb.x0 += dx
        bb.x1 += dx
        leg.set_bbox_to_anchor(bb, fp.axes.flatten()[-1].transAxes)
        fp.fig.tight_layout(pad=0.5)
        plt.savefig(
            "notebooks/results/charts/1_iter_{}.pdf".format(approach),
            dpi=300,
            bbox_inches="tight",
        )


def plot_by_iter(df):
    df = df[df["curr-iter"] == df["num-iter"]]
    df = df[df["num-iter"] > 0]
    df = df[df["workload"] != 0]

    for query_st in ["opt-l", "opt-rl", "unc-l", "unc-rl"]:
        print(query_st)
        df_query = df[df["query-st"] == query_st]

        fp = sns.FacetGrid(
            hue="workload",
            height=3,
            aspect=1.3,
            data=df_query,
            col="dataset-test",
            col_order=["S-MARQUES", "S-ISRI-OCR", "S-CDIP"],
            sharey=False,
            legend_out=True,
        )
        fp = fp.map(
            sns.lineplot,
            "num-iter",
            "accuracy",
            ci=None,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgewidth=0.1,
        )
        fp.set_axis_labels("$n_{iter}$", "accuracy (\%)", fontsize=26)
        fp = fp.add_legend(title="$\\alpha_{load}$", fontsize=22, labelspacing=0.25)

        # adjust each axis
        for ax in fp.axes.flatten():
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            dataset = ax.title.get_text()
            dataset = dataset.replace("dataset-test = ", "")
            dataset = map_dataset[dataset]
            ax.set_title(dataset, fontsize=24)
            ax.set_xticks([1, 2, 3])
            for text in ax.get_xticklabels():
                text.set_fontsize(20)
                text.set_usetex(False)
            for text in ax.get_yticklabels():
                text.set_fontsize(20)
                text.set_usetex(False)

        leg = fp._legend
        leg.get_title().set_fontsize(20)
        for text in leg.get_texts():
            text.set_fontsize(20)
            text.set_text(map_wload[text.get_text()])

        bb = leg.get_bbox_to_anchor().transformed(
            fp.axes.flatten()[-1].transAxes.inverted()
        )
        dx = 0.1
        dy = 0
        bb.y0 += dy
        bb.y1 += dy
        bb.x0 += dx
        bb.x1 += dx
        leg.set_bbox_to_anchor(bb, fp.axes.flatten()[-1].transAxes)
        fp.fig.tight_layout(pad=0.75)
        plt.savefig(
            "notebooks/results/charts/by_iter_{}.pdf".format(query_st),
            dpi=300,
            bbox_inches="tight",
        )
