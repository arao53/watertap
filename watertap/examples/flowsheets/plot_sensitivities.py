import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def contour_figure(
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

    ax.set_xlabel("Electricity price [$/kWh]")
    ax.set_ylabel("Utilization factor [-]")
    ax.set_title("LCOW [$/m3]")
    ax.set_ylim(top=max(y))
    plt.show()
    return fig, ax


def line_sensitivities(path_to_results):
    df = pd.read_csv(path_to_results)
    column_names = df.columns.to_numpy()

    x_col = 0

    xlabel = column_names[x_col]
    ylabel1 = column_names[1]
    ylabel2 = column_names[2]
    ylabel3 = column_names[3]
    ylabel4 = column_names[4]

    x = df[xlabel].values
    y1 = df[ylabel1].values
    y2 = df[ylabel2].values
    y3 = df[ylabel3].values
    y4 = df[ylabel4].values

    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Electricity price sensitivity")
    ax[0, 0].plot(x, y1)
    ax[0, 0].set_xlabel(xlabel)
    ax[0, 0].set_ylabel(ylabel1 + "[$/m3]")

    ax[0, 1].plot(x, y2 * 3600)
    ax[0, 1].set_xlabel(xlabel)
    ax[0, 1].set_ylabel(ylabel2 + "[m3/hr]")

    ax[1, 0].plot(x, y3)
    ax[1, 0].set_xlabel(xlabel)
    ax[1, 0].set_ylabel(ylabel3 + "[-]")

    ax[1, 1].plot(x, y4)
    ax[1, 1].set_xlabel(xlabel)
    ax[1, 1].set_ylabel(ylabel4 + "[bar]")
    return fig, ax


def utilization_sensitivity(
    path_to_results,
    x_col=None,
    y_col=None,
    iso_col=None,
    isolines=None,
):
    x_col = (1,)
    y_col = (2,)
    iso_col = (0,)
    isolines = ([0.0, 0.05, 0.1, 0.15],)

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
    ax.set_ylim([0.2, 1.2])

    ax.set_ylabel("LCOW [$/m3]")
    ax.set_xlabel("Utilization factor [-]")
    plt.legend()

    return fig, ax


def electricity_sensitivity(path_to_results):
    x_col = (0,)
    y_col = (2,)
    iso_col = (1,)
    isolines = ([0.5, 0.7, 0.9, 1],)

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
    ax.set_ylim([0.2, 1.2])

    ax.set_ylabel("LCOW [$/m3]")
    ax.set_xlabel("Electricity Cost [$/kWh]")
    plt.legend()

    return fig, ax


def bar_plot(path_to_results):
    discrete_entries = 6
    nx = 4
    n_output = 1

    df = pd.read_csv(path_to_results)
    cols = df.columns.to_numpy()
    arr = df.to_numpy()

    # NOTE: The discrete variable must be the first col and the output must be the last col
    # reshape the array for easier indexing
    resized_arr_shape = np.array([discrete_entries])
    for i in range(len(cols) - 1 - n_output):  # 1 for discrete vars
        resized_arr_shape = np.append(resized_arr_shape, nx)
    arr_r = np.reshape(arr, resized_arr_shape)
    arr_r[:, 0, 0] = arr_r[:, 0, 0] * 300

    plotting_arr = np.ndarray([3, discrete_entries])  # discrete_var, range, min
    for j in range(discrete_entries):
        discrete_var = arr_r[j, 0, 0]
        min_output = np.min(arr_r[j, :, -1])
        range_output = np.max(arr_r[j, :, -1]) - min_output
        plotting_arr[:, j] = [discrete_var, range_output, min_output]

    fig, ax = plt.subplots()
    ax.bar(
        x=plotting_arr[0, :],
        height=plotting_arr[1, :],
        width=0.5 * np.min(plotting_arr[0, :]),
        bottom=plotting_arr[2, :],
        align="center",
        edgecolor="black",
        alpha=0.5,
        rasterized=True,
    )
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[-1])
    ax.set_xticks(plotting_arr[0, :])

    ax.set_ylim([0, 3.25])

    ax.set_ylabel("LCOW [$/m3]")
    ax.set_xlabel("Base capital cost [$/m3/day]")
    ax.text(
        1200,
        2.25,
        "Worst case:\n Utilization Factor= 50%\n Electricity= 0.25$/kWh",
        ha="center",
    )
    ax.text(
        3200,
        0.2,
        "Ideal case:\n Utilization Factor= 100%\n Electricity= 0.0$/kWh",
        ha="center",
    )

    return fig, ax


if __name__ == "__main__":
    current_dir = os.getcwd()
    file_of_interest = "RO_with_energy_recovery\\sensitivity_2.csv"
    path = os.path.join(current_dir, file_of_interest)

    # # line sensitivities
    # fig, ax = line_sensitivities(path)
    #
    # # contour plot
    # fig1, ax1 = contour_figure(path, plot_type="contourf", levels=10)
    #
    #
    # # multi line plots
    # fig2, ax2 = utilization_sensitivity(path)
    # fig3, ax3 = electricity_sensitivity(path)
    #
    #
    # # bar plot
    # fig4, ax4 = bar_plot(path)
