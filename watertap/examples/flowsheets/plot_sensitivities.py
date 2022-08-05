import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from idaes.core.util.misc import StrEnum


class plot_type(StrEnum):
    contour = "contour"
    line = "line"
    multiline = "multiline"


def plot_results(path_to_results, type, title=None, xlabel=None, ylabel=None):

    if type == "contour":
        df = pd.read_csv(path_to_results)
        column_names = df.columns.to_numpy()
        df = df.pivot(column_names[0], column_names[1], column_names[2])
        y = df.columns.values
        x = df.index.values
        Z = df.values.T
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z, levels=20, cmap="YlGnBu_r")

        if xlabel is None:
            xlabel = column_names[0]
        if ylabel is None:
            ylabel = column_names[1]
        if title is None:
            title = column_names[2]

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(top=max(y))
        cbar = fig.colorbar(cs)
        plt.show()
        return fig, ax

    #
    # elif type == "line":
    #     xlabel = sweep_params.keys()[0]
    #     X = global_results[:, 0]
    #     Y = global_results[:, 1]
    #     fig, ax = plt.subplots()
    #     ax.plot(X, Y)
    #     ax.set_xlabel(xlabel)


if __name__ == "__main__":
    path = "C:\\Users\\aksha\\GitStuff\\WaterTap3_KL\\watertap\\watertap\\examples\\flowsheets\\RO_with_energy_recovery\\sensitivity_1.csv"
    fig, ax = plot_results(path, type="contour")
