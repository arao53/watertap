import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyomo.environ import Constraint
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
    outputs["membrane_area"] = m.fs.RO.area
    outputs["SCI"] = m.fs.costing.specific_electrical_carbon_intensity
    outputs["annual_capex"] = m.fs.costing.annual_investment
    outputs["annual_opex"] = m.fs.costing.total_operating_cost
    outputs["baseline_cost"] = m.fs.costing.baseline_daily_cost
    outputs["electricity_price"] = m.fs.costing.electricity_cost

    return outputs, optimize_kwargs, opt_function


def run_analysis(case_num, nx, interpolate_nan_outputs):
    m = swro.main(
        erd_type=swro.ERDtype.pump_as_turbine,
        variable_efficiency=swro.VariableEfficiency.flow,
    )

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
        m.fs.RO.area.fix()
        m.fs.P1.bep_flow.fix()
        m.fs.P1.flow_ratio[0].unfix()
        m.fs.P1.flow_ratio[0].setub(1)
        m.fs.P1.control_volume.properties_out[0].pressure.unfix()
        m.fs.RO.recovery_mass_phase_comp[0.0, "Liq", "H2O"].unfix()
        m.fs.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", "NaCl"].setub(0.5)
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].unfix()
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(35)

        sweep_params["electricity_price"] = LinearSample(
            m.fs.costing.electricity_cost, 0, 0.5, nx
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
    elif case_num == 4:
        sweep_params["recovery_ratio"] = LinearSample(
            m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"], 0.3, 0.6, nx
        )
        sweep_params["co2_intensity"] = LinearSample(
            m.fs.costing.electrical_carbon_intensity, 0, 1, nx
        )

    elif case_num == 5:
        m.fs.RO.area.fix()
        m.fs.P1.bep_flow.fix()
        m.fs.P1.flow_ratio[0].unfix()
        m.fs.P1.flow_ratio[0].setub(1)
        m.fs.P1.control_volume.properties_out[0].pressure.unfix()
        m.fs.RO.recovery_mass_phase_comp[0.0, "Liq", "H2O"].unfix()
        m.fs.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", "NaCl"].setub(0.5)
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].unfix()
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(35)
        m.fs.costing.electricity_cost.unfix()
        m.fs.costing.cost_constraint = Constraint(
            expr=m.fs.costing.annual_investment == m.fs.costing.total_operating_cost,
            doc="Solves for the breakeven between investment and operating cost",
        )

        sweep_params["investment_factor"] = LinearSample(
            m.fs.costing.factor_total_investment, 1, 5, nx
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


def main(case_num=5, nx=21, interpolate_nan_outputs=False):
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
