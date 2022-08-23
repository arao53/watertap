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
    outputs["permeate_flowrate"] = m.fs.product.properties[0].flow_vol_phase["Liq"]
    outputs["recovery_rate"] = m.fs.RO.recovery_mass_phase_comp[0.0, "Liq", "H2O"]
    outputs["pressure"] = m.fs.P1.control_volume.properties_out[0].pressure * 1e-5

    return outputs, optimize_kwargs, opt_function


def run_analysis(case_num, nx, interpolate_nan_outputs):
    m = swro.main()

    outputs, optimize_kwargs, opt_function = set_up_sensitivity(m)

    sweep_params = {}
    if case_num == 1:
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0.0, 0.25, nx
        )
        sweep_params["utilization_factor"] = LinearSample(
            m.fs.costing.utilization_factor, 0.25, 1, nx
        )
    elif case_num == 2:
        # m.fs.costing.factor_total_investment(3)
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0, 2, nx
        )
    elif case_num == 3:
        sweep_params["investment_factor"] = LinearSample(
            m.fs.costing.factor_total_investment, 2, 12, 6
        )
        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0.0, 0.25, nx
        )
        sweep_params["utilization_factor"] = LinearSample(
            m.fs.costing.utilization_factor, 0.5, 1, nx
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
