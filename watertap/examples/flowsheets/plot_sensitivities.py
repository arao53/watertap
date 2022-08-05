import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(
    path_to_results,
    plot_type,
    levels=20,
    title=None,
    xlabel=None,
    ylabel=None,
    x_col=None,
    y_col=None,
    iso_col=None,
    isolines=None,
):

    if plot_type == "contourf" or plot_type == "contour":
        df = pd.read_csv(path_to_results)
        column_names = df.columns.to_numpy()
        df = df.pivot(column_names[0], column_names[1], column_names[2])
        y = df.columns.values
        x = df.index.values
        Z = df.values.T
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()

        if plot_type == "contourf":
            cs = ax.contourf(X, Y, Z, levels=levels, cmap="YlGnBu_r")
            cbar = fig.colorbar(cs)
        else:
            cs = ax.contour(X, Y, Z, levels=levels, cmap="YlGnBu_r")
            clbl = ax.clabel(cs, colors="black")

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
        plt.show()

    elif plot_type == "line_simple":
        df = pd.read_csv(path_to_results)
        column_names = df.columns.to_numpy()

        xlabel = column_names[0]
        ylabel = column_names[1]

        x = df[xlabel].values
        y = df[xlabel].values

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif plot_type == "line_multi":
        if x_col is None or isolines is None:
            raise ValueError(
                "Must include x_col, y_col, iso_col and isolines for line_multi plot_type"
            )

        df = pd.read_csv(path_to_results)
        column_names = df.columns.to_numpy()

        xlabel = column_names[x_col]
        ylabel = column_names[y_col]
        iso_variable = column_names[iso_col]

        df = df.pivot(xlabel, iso_variable, ylabel)

        fig, ax = plt.subplots()

        x = df.index.values

        for i in isolines:
            col_num = np.where(df.columns == i)
            y = df.values.T[col_num].T
            ax.plot(x, y, label=iso_variable + "=" + str(i))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.legend()

    elif plot_type == "floating_bar":
        df = pd.read_csv(path_to_results)

    else:
        raise ValueError("plot_type not recognized")

    return fig, ax


if __name__ == "__main__":
    current_dir = os.getcwd()
    file_of_interest = "RO_with_energy_recovery\\sensitivity_1.csv"
    path = os.path.join(current_dir, file_of_interest)
    # fig, ax = plot_results(path, plot_type="contour", levels=5)
    fig, ax = plot_results(
        path,
        plot_type="line_multi",
        x_col=1,
        y_col=2,
        iso_col=0,
        isolines=[0.05, 0.075, 0.1, 0.125],
    )
