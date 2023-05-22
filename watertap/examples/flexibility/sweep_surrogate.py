# imports
from watertap.examples.flowsheets.RO_with_energy_recovery import RO_with_energy_recovery as swro
from watertap.tools.parameter_sweep import LinearSample, parameter_sweep
from idaes.core.solvers import get_solver

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Expression,
    exp,
    Objective,
    minimize, 
    )

# build parameter sweep

def get_metrics(m, num_samples):
    sweep_params = {}

    sweep_params["recovery"] = LinearSample(
        m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"],
        0.05, 0.625, num_samples
    )

    outputs = {}

    outputs["hwp"] = m.fs.costing.annual_water_production / (365.25 * 24)
    outputs["sec"] = m.fs.costing.specific_energy_consumption

    return sweep_params, outputs


def run_case(num_samples=21, filepath = None):
    m = swro.build()
    swro.set_operating_conditions(m,water_recovery=0.65)
    swro.initialize_system(m)
    m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.P1.control_volume.properties_out[0].pressure.unfix()
    m.fs.RO.area.fix(120)
    swro.solve(m)

    # m = unfix_dof(m)

    params, outputs = get_metrics(m, num_samples)
    
    global_results = parameter_sweep(
        m,
        params,
        outputs,
        csv_results_file_name=filepath,
        optimize_function=swro.solve,
        interpolate_nan_outputs=False,
    )

    return m, global_results

def fit_exponential_quadratic(input, output):
    
    n = len(input)
    
    # create a matrix of basis functions on recovery
    tmp = ConcreteModel()
    tmp.idx = range(len(input))

    # create some fitting parameters 
    tmp.a = Var(initialize=0.1,
            bounds=(None,None))

    tmp.b = Var(initialize=0.1,
            bounds=(None,None))
    tmp.c = Var(initialize=0.1,
            bounds = (None,None))

    tmp.d = Var(initialize=0.1,
                bounds=(None,None))


    tmp.y1 = Var(tmp.idx,
            initialize=0.1,)
    tmp.y2 = Var(tmp.idx,
            initialize=0.1,)
    
    # create basis functions using the fitting parameters
    tmp.con1 = Constraint(tmp.idx,
        expr = [tmp.y1[i] == tmp.a * exp(-tmp.b * input[i]) for i in tmp.idx])
    tmp.con2 = Constraint(tmp.idx,
        expr = [tmp.y2[i] == tmp.c * (input[i])**2 + tmp.d for i in tmp.idx])

    tmp.y = Expression(tmp.idx,
                    expr = [tmp.y1[i] + tmp.y2[i] for i in tmp.idx])

    # # create the objective function
    tmp.obj = Objective(
        expr = sum((tmp.y1[i] + tmp.y2[i] - output[i])**2 for i in tmp.idx),
        sense= minimize)
    
    # solve with ipopt

    solver = get_solver()
    solver.solve(tmp, tee = False)

    # check fit
    if tmp.obj()/n > 1e-2:
        raise ValueError("Fit is not good enough")
     
    return [tmp.a.value, tmp.b.value, tmp.c.value, tmp.d.value], tmp

if __name__ == "main":
    # create the dataset
    m, results = run_case(num_samples=21, filepath = None)
    
    recovery = results[0][:,0]
    sec = results[0][:,2]

    # fit the data 
    fit_params, tmp = fit_exponential_quadratic(recovery, sec)