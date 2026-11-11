"""Run the CASE-PENETRATION parameter study in one job (cases are independent)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_case as rc

CASES = {
 "base":   dict(width_grains=24, depth_grains=10, radius_over_d=4.0, mu=0.40, seed=20261112),
 "coarse": dict(width_grains=24, depth_grains=10, radius_over_d=3.0, mu=0.40, seed=7),
 "mu030":  dict(width_grains=24, depth_grains=10, radius_over_d=4.0, mu=0.30, seed=11),
 "slow":   dict(width_grains=24, depth_grains=10, radius_over_d=4.0, mu=0.40, seed=20261112,
                dz_over_d=0.25, relax=300),
}

def main(which):
    out = {}
    for name in which:
        kw = dict(grain_d=0.004, relax=300, substeps=3, settle=1200,
                  penetrate_over_r=2.0, fit_from_over_r=1.1, damping=200.0,
                  phi_request=0.60, packing_gap=0.02, dt_factor=1.0,
                  mu_rolling=0.05)
        kw.update(CASES[name])
        ns = rc.argparse.Namespace(**kw, out="runs/case_penetration/" + name)
        print("=" * 60, name, flush=True)
        try:
            res = rc.run_case(ns)
        except Exception as exc:  # keep the rest of the study alive
            print("CASE FAILED", name, repr(exc), flush=True)
            out[name] = dict(failed=repr(exc))
            continue
        os.makedirs(ns.out, exist_ok=True)
        json.dump(res, open(os.path.join(ns.out, "result.json"), "w"), indent=2)
        out[name] = dict(alpha_fit=res["result"]["alpha_fit"],
                         err=res["result"]["err_vs_ref"],
                         phi=res["parameters"]["packing_fraction"],
                         n=res["parameters"]["n_grains"],
                         t_s=res["timing"]["total_s"])
        print(name, out[name], flush=True)
    json.dump(out, open("runs/case_penetration/summary.json", "w"), indent=2)

if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
