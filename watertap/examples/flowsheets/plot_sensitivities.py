import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.rc("font", size=10)
plt.rc("axes", titlesize=10)
scaling_obj = 1
scaling_factor = 1


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
    y = df.columns.values * 100
    x = df.index.values
    Z = df.values.T
    X, Y = np.meshgrid(x, y)

    max_val = np.max(Z)

    # norm_Z = 100*(1 - Z/max_val)
    base = 0.5
    norm_Z = 100 * (Z - base) / base
    fig, ax = plt.subplots()

    if plot_type == "contourf":
        cs = ax.contourf(X, Y, norm_Z, levels=levels, cmap="YlGnBu_r", vmin=0, vmax=200)
        cbar = fig.colorbar(cs)
        # cs = ax.contour(X, Y, norm_Z, levels=isolines, cmap="twilight")
        # clbl = ax.clabel(cs)

    else:
        cs = ax.contour(X, Y, norm_Z, levels=levels, cmap="YlGnBu_r")
        clbl = ax.clabel(cs, colors="black")

    if xlabel is None:
        xlabel = column_names[0]
    if ylabel is None:
        ylabel = column_names[1]
    if title is None:
        title = column_names[2]

    ax.set_xlabel("Electricity price [$/kWh]")
    ax.set_ylabel("Utilization factor [%]")
    ax.set_title("SWRO: LCOW Compared to baseline 0.5$/m3 [%]")
    ax.text(0.195, 28, " LCOW >\n1.5$/m3", color="black", fontsize="large")
    ax.text(0.008, 90, " LCOW <\n0.5$/m3", color="white", fontsize="large")
    ax.set_ylim(top=max(y))
    plt.show()
    return fig, ax


def carbon_intensity_contour(
    path_to_results,
    plot_type,
    levels=20,
    title=None,
    xlabel=None,
    ylabel=None,
    xcol=0,
    ycol=1,
    zcol=2,
    iso_col=None,
    isolines=None,
):
    df = pd.read_csv(path_to_results)
    column_names = df.columns.to_numpy()
    df = df.pivot(column_names[xcol], column_names[ycol], column_names[zcol])
    y = df.columns.values
    x = df.index.values * 100
    Z = df.values.T
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    if plot_type == "contourf":
        cs = ax.contourf(X, Y, Z, levels=levels, cmap="Reds")
        cbar = fig.colorbar(cs)
        # cs = ax.contour(X, Y, norm_Z, levels=isolines, cmap="twilight")
        # clbl = ax.clabel(cs)

    else:
        cs = ax.contour(X, Y, Z, levels=levels, cmap="YlGnBu_r")
        clbl = ax.clabel(cs, colors="black")

    if xlabel is None:
        xlabel = column_names[0]
    if ylabel is None:
        ylabel = column_names[1]
    if title is None:
        title = column_names[2]

    ax.set_xlabel("RO Recovery Ratio [%]")
    ax.set_ylabel("Grid Carbon Intensity [kg/kWh]")
    ax.set_title("Specific Carbon Intensity [kgCO2eq/m3]")
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
    ylabel5 = column_names[5]
    ylabel6 = column_names[6]

    x = df[xlabel].values
    y1 = df[ylabel1].values
    y2 = df[ylabel2].values
    y3 = df[ylabel3].values
    y4 = df[ylabel4].values
    y5 = df[ylabel5].values
    y6 = df[ylabel6].values

    fig, ax = plt.subplots(3, 2)
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

    ax[2, 0].plot(x, y5)
    ax[2, 0].set_xlabel(xlabel)
    ax[2, 0].set_ylabel(ylabel5 + "[m2]")

    ax[2, 1].plot(x, y6)
    ax[2, 1].set_xlabel(xlabel)
    ax[2, 1].set_ylabel(ylabel6 + "[kgCO2eq/m3]")
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


def electricity_price_breakdown(path_to_results):
    df = pd.read_csv(path_to_results)
    column_names = df.columns.to_numpy()

    xlabel = "# electricity_price"
    y1_label = "annual_capex"
    y2_label = "annual_opex"

    fig, ax = plt.subplots()

    x = df[xlabel].values
    y1 = df[y1_label].values
    y2 = df[y2_label].values

    ax.plot(x, y1, label=y1_label)
    ax.plot(x, y2, label=y2_label)

    ax.set_xlabel("Electricity Price $/kWh")
    ax.set_ylabel("Cost $/year")

    plt.legend()

    return fig, ax


def breakeven_electricity_price(path_to_results):
    df = pd.read_csv(path_to_results)
    column_names = df.columns.to_numpy()

    xlabel = "baseline_cost"
    y1_label = "electricity_price"

    fig, ax = plt.subplots()

    x = df[xlabel].values
    y1 = df[y1_label].values

    ax.plot(x, y1)
    ax.text(500, 0.12, "Flexiblity may be optimal")
    ax.text(1400, 0.03, "Fixed is optimal")

    ax.set_xlabel("Baseline cost $/m3/day")
    ax.set_ylabel("Electricity price $/m3")
    ax.set_title("Breakeven for reduced operations")

    return fig, ax


if __name__ == "__main__":
    current_dir = os.getcwd()
    file_of_interest = "RO_with_energy_recovery\\sensitivity_5.csv"
    path = os.path.join(current_dir, file_of_interest)

    # line sensitivities
    # fig, ax = line_sensitivities(path)

    fig, ax = breakeven_electricity_price(file_of_interest)

    # contour plot
    # fig1, ax1 = contour_figure(
    #     path, plot_type="contourf", levels=48, isolines=[0, 100, 200]
    # )
    #
    #
    # # multi line plots
    # fig2, ax2 = utilization_sensitivity(path)
    # fig3, ax3 = electricity_sensitivity(path)
    #
    #
    # # bar plot
    # fig4, ax4 = bar_plot(path)

    # carbon intensity contour
    # fig5, ax5 = carbon_intensity_contour(
    #     path,
    #     plot_type="contourf",
    #     zcol=3,
    # )
