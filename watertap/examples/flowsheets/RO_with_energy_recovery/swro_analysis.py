import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from watertap.tools.parameter_sweep import _init_mpi, LinearSample, parameter_sweep
import watertap.examples.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery as swro


def set_up_sensitivity(m):
    outputs = {}
    optimize_kwargs = {"check_termination": False}  # None
    opt_function = swro.solve

    # create outputs
    outputs["LCOW"] = m.fs.costing.LCOW

    return outputs, optimize_kwargs, opt_function


def run_analysis(case_num=2, nx=20, interpolate_nan_outputs=False):
    m = swro.main()

    outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

    sweep_params = {}
    if case_num == 1:
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0.0, 0.25, nx
        )
        sweep_params["utilization_factor"] = LinearSample(
            m.fs.costing.utilization_factor, 0.5, 1, nx
        )
    elif case_num == 2:
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0.01, 0.5, nx
        )
    elif case_num == 3:
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_base_cost, 0.01, 0.5, nx
        )
    else:
        raise ValueError("case_num = %d not recognized." % (case_num))

    output_filename = "sensitivity_" + str(case_num) + ".csv"

    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=opt_function,
        optimize_kwargs=optimize_kwargs,
        interpolate_nan_outputs=interpolate_nan_outputs,
    )

    return global_results, sweep_params, m


def plot_results(global_results, sweep_params):
    if len(sweep_params.keys()) == 2:
        xlabel, ylabel = sweep_params.keys()
        df = pd.DataFrame(global_results, columns=[xlabel, ylabel, "results"])
        df = df.pivot(xlabel, ylabel, "results")
        y = df.columns.values * 100  # convert to %
        x = df.index.values
        Z = df.values.T
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, Z, levels=20, cmap="YlGnBu_r")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(top=max(y))
        ax.clabel(cs, inline=True, fontsize=10)
        # cbar = fig.colorbar(cs)

    elif len(sweep_params.keys()) == 1:
        xlabel = sweep_params.keys()[0]
        X = global_results[:, 0]
        Y = global_results[:, 1]
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.set_xlabel(xlabel)

    else:
        raise NotImplementedError(
            "Plot types with >2 indep variables have not been implemented"
        )

    return fig, ax


def main(case_num=1, nx=21, interpolate_nan_outputs=False):
    # when from the command line
    case_num = int(case_num)
    nx = int(nx)
    interpolate_nan_outputs = bool(interpolate_nan_outputs)

    # Start MPI communicator
    comm, rank, num_procs = _init_mpi()

    tic = time.time()
    global_results, sweep_params, m = run_analysis(
        case_num, nx, interpolate_nan_outputs
    )
    print(global_results)
    toc = time.time()

    if rank == 0:
        total_samples = 1

        for k, v in sweep_params.items():
            total_samples *= v.num_samples

        print("Finished case_num = %d." % (case_num))
        print(
            "Processed %d swept parameters comprising %d total points."
            % (len(sweep_params), total_samples)
        )
        print("Elapsed time = %.1f s." % (toc - tic))
    return global_results, sweep_params, m


if __name__ == "__main__":
    global_results, sweep_params, m = main(*sys.argv[1:])
    fig, ax = plot_results(global_results, sweep_params)
