from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

from FitnessFunctions import Jump, LeadingOnes, OneMax
from TerminationCriterion import TerminationCriterion
from cGA import cGA
from sigcGA import sigcGA
from onePlusOneEA import onePlusOneEA


# -------------------- Config objects --------------------

@dataclass(frozen=True)
class AlgoConfig:
    name: str                      # "cGA" | "sigcGA" | "onePlusOneEA"
    params: Dict[str, Any]         # {"k": ...} or {"eps": ...} or {}

@dataclass(frozen=True)
class ExperimentConfig:
    fitness_name: str              # "OneMax" | "LeadingOnes" | "Jump"
    n: int
    budget: int
    jump_k: Optional[int]
    algo: AlgoConfig

@dataclass
class RunResult:
    fitness_name: str
    n: int
    budget: int
    algo_name: str
    algo_params: str               # JSON string
    iterations: int
    evaluations: int               # IMPORTANT: call_count


# -------------------- Helpers --------------------

def parse_grid(s: str, kind: str) -> list[float]:
    """
    Accepts:
      - comma list: "12,10,8"
      - range: "start:stop:step" (inclusive-ish)
    Returns floats (cast to int where needed).
    """
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad {kind} grid '{s}'. Use 'a:b:c' or 'v1,v2,...'")
        start, stop, step = map(float, parts)
        if step == 0:
            raise ValueError("step cannot be 0")
        vals = []
        x = start
        if step > 0:
            while x <= stop + 1e-12:
                vals.append(x)
                x += step
        else:
            while x >= stop - 1e-12:
                vals.append(x)
                x += step
        return vals
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_fitness(name: str, jump_k: Optional[int]):
    if name == "OneMax":
        return OneMax()
    if name == "LeadingOnes":
        return LeadingOnes()
    if name == "Jump":
        if jump_k is None:
            raise ValueError("Jump requires --jump_k")
        return Jump(jump_k)  # adapt if your Jump signature differs
    raise ValueError(f"Unknown fitness: {name}")


def make_termination(fitness, n: int, budget: int):
    return TerminationCriterion(fitness=fitness, n=n, budget=budget)


# -------------------- Single run --------------------

def run_once(cfg: ExperimentConfig, seed: int) -> RunResult:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    fitness = make_fitness(cfg.fitness_name, cfg.jump_k)
    termination = make_termination(fitness, cfg.n, cfg.budget)

    algo = cfg.algo.name
    params = cfg.algo.params

    if algo == "onePlusOneEA":
        _p, t = onePlusOneEA(cfg.n, fitness, termination, py_rng, np_rng)
    elif algo == "cGA":
        k = int(params["k"])
        _p, t = cGA(cfg.n, k, fitness, termination, py_rng, np_rng)
    elif algo == "sigcGA":
        eps = float(params["eps"])
        _p, t = sigcGA(cfg.n, eps, fitness, termination, py_rng, np_rng)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return RunResult(
        fitness_name=cfg.fitness_name,
        n=cfg.n,
        budget=cfg.budget,
        algo_name=algo,
        algo_params=json.dumps(params, sort_keys=True),
        iterations=int(t),
        evaluations=int(fitness.call_count),
    )


# -------------------- Multi runs --------------------

def iter_seeds(base_seed: int, count: int) -> Iterable[int]:
    for i in range(count):
        yield base_seed + i


def run_repetitions(cfg: ExperimentConfig, repetitions: int, base_seed: int) -> list[RunResult]:
    return [run_once(cfg, seed) for seed in iter_seeds(base_seed, repetitions)]


def run_grid(configs: list[ExperimentConfig], repetitions: int, base_seed: int) -> list[RunResult]:
    all_results: list[RunResult] = []
    seed_stride = 10_000
    for idx, cfg in enumerate(configs):
        cfg_base = base_seed + idx * seed_stride
        all_results.extend(run_repetitions(cfg, repetitions, cfg_base))
    return all_results


# -------------------- Output / Aggregation --------------------

def write_csv(path: str, results: list[RunResult]) -> None:
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def mean_evaluations(results: list[RunResult]) -> float:
    return float(np.mean([r.evaluations for r in results])) if results else float("nan")


def group_mean_evals_by_param(results: list[RunResult], param_name: str) -> dict[float, float]:
    buckets: dict[float, list[int]] = {}
    for r in results:
        params = json.loads(r.algo_params)
        if param_name not in params:
            continue
        x = float(params[param_name])
        buckets.setdefault(x, []).append(int(r.evaluations))
    return {x: float(np.mean(v)) for x, v in sorted(buckets.items())}


def group_mean_evals_by_n(results: list[RunResult]) -> dict[int, float]:
    buckets: dict[int, list[int]] = {}
    for r in results:
        buckets.setdefault(int(r.n), []).append(int(r.evaluations))
    return {n: float(np.mean(v)) for n, v in sorted(buckets.items())}


def plot_sweep(
    out_png: str,
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    title: str,
    baseline_mean_eval: float,
    budget: int,
):
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.axhline(baseline_mean_eval, linestyle="--", label="(1+1)EA mean evaluations")
    plt.axhline(budget, linestyle="--", label=f"Budget (evaluations={budget})")
    plt.xlabel(x_label)
    plt.ylabel("Mean evaluations over REPS")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------- CLI main --------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["sweep_param", "sweep_n"], required=True,
                    help="sweep_param: sweep eps (sigcGA) or k (cGA). sweep_n: sweep n with fixed algo param.")

    ap.add_argument("--algo", choices=["sigcGA", "cGA"], required=True)
    ap.add_argument("--fitness", choices=["OneMax", "LeadingOnes", "Jump"], required=True)

    ap.add_argument("--n", type=int, default=None, help="Used in sweep_param.")
    ap.add_argument("--n_grid", type=str, default=None, help="Used in sweep_n. e.g. '50:500:50' or '50,100,200'")

    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--jump_k", type=int, default=None)

    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--base_seed", type=int, default=42)

    # Param grids (mode=sweep_param)
    ap.add_argument("--eps_grid", type=str, default="12,10,8,6,4,2")
    ap.add_argument("--k_grid", type=str, default="50,100,150,200")

    # Fixed param (mode=sweep_n)
    ap.add_argument("--eps", type=float, default=None, help="Fixed eps for sigcGA in sweep_n")
    ap.add_argument("--k", type=int, default=None, help="Fixed k for cGA in sweep_n")

    ap.add_argument("--out_csv", type=str, default="results.csv")
    ap.add_argument("--out_plot", type=str, default=None)

    args = ap.parse_args()

    if args.fitness == "Jump" and args.jump_k is None:
        raise SystemExit("Error: fitness=Jump requires --jump_k")

    # baseline (1+1)EA always included so you can draw the horizontal mean line
    def baseline_cfg(n_val: int) -> ExperimentConfig:
        return ExperimentConfig(
            fitness_name=args.fitness,
            n=n_val,
            budget=args.budget,
            jump_k=args.jump_k,
            algo=AlgoConfig(name="onePlusOneEA", params={}),
        )

    configs: list[ExperimentConfig] = []

    if args.mode == "sweep_param":
        if args.n is None:
            raise SystemExit("Error: mode=sweep_param requires --n")
        n_val = int(args.n)

        configs.append(baseline_cfg(n_val))

        if args.algo == "sigcGA":
            for eps in parse_grid(args.eps_grid, "eps"):
                configs.append(
                    ExperimentConfig(
                        fitness_name=args.fitness,
                        n=n_val,
                        budget=args.budget,
                        jump_k=args.jump_k,
                        algo=AlgoConfig(name="sigcGA", params={"eps": float(eps)}),
                    )
                )
        else:  # cGA
            for k in parse_grid(args.k_grid, "k"):
                configs.append(
                    ExperimentConfig(
                        fitness_name=args.fitness,
                        n=n_val,
                        budget=args.budget,
                        jump_k=args.jump_k,
                        algo=AlgoConfig(name="cGA", params={"k": int(round(k))}),
                    )
                )

    else:  # mode == "sweep_n"
        if args.n_grid is None:
            raise SystemExit("Error: mode=sweep_n requires --n_grid")
        n_vals = [int(round(x)) for x in parse_grid(args.n_grid, "n")]

        if args.algo == "sigcGA":
            if args.eps is None:
                raise SystemExit("Error: mode=sweep_n with sigcGA requires --eps (fixed)")
            fixed_params = {"eps": float(args.eps)}
            algo_cfg = AlgoConfig(name="sigcGA", params=fixed_params)
        else:
            if args.k is None:
                raise SystemExit("Error: mode=sweep_n with cGA requires --k (fixed)")
            fixed_params = {"k": int(args.k)}
            algo_cfg = AlgoConfig(name="cGA", params=fixed_params)

        # include baseline + algo for each n
        for n_val in n_vals:
            configs.append(baseline_cfg(n_val))
            configs.append(
                ExperimentConfig(
                    fitness_name=args.fitness,
                    n=int(n_val),
                    budget=args.budget,
                    jump_k=args.jump_k,
                    algo=algo_cfg,
                )
            )

    results = run_grid(configs, repetitions=args.reps, base_seed=args.base_seed)
    write_csv(args.out_csv, results)
    print(f"Wrote {len(results)} rows to {args.out_csv}")

    if args.out_plot is not None:
        if args.mode == "sweep_param":
            # baseline mean eval
            base_runs = [r for r in results if r.algo_name == "onePlusOneEA"]
            base_mean = mean_evaluations(base_runs)

            if args.algo == "sigcGA":
                sweep_runs = [r for r in results if r.algo_name == "sigcGA"]
                means = group_mean_evals_by_param(sweep_runs, "eps")
                xs = list(means.keys())
                ys = [means[x] for x in xs]
                plot_sweep(
                    out_png=args.out_plot,
                    x_values=xs,
                    y_values=ys,
                    x_label="epsilon (eps)",
                    title=f"sigcGA sweep (mean evaluations) on {args.fitness} (n={args.n}, budget={args.budget})",
                    baseline_mean_eval=base_mean,
                    budget=args.budget,
                )
            else:
                sweep_runs = [r for r in results if r.algo_name == "cGA"]
                means = group_mean_evals_by_param(sweep_runs, "k")
                xs = list(means.keys())
                ys = [means[x] for x in xs]
                plot_sweep(
                    out_png=args.out_plot,
                    x_values=xs,
                    y_values=ys,
                    x_label="population K",
                    title=f"cGA sweep (mean evaluations) on {args.fitness} (n={args.n}, budget={args.budget})",
                    baseline_mean_eval=base_mean,
                    budget=args.budget,
                )

        else:  # sweep_n
            # baseline per n and algo per n
            algo_name = args.algo
            base_by_n = group_mean_evals_by_n([r for r in results if r.algo_name == "onePlusOneEA"])
            algo_by_n = group_mean_evals_by_n([r for r in results if r.algo_name == algo_name])

            ns = sorted(set(base_by_n.keys()) & set(algo_by_n.keys()))
            xs = [float(n) for n in ns]
            ys = [algo_by_n[n] for n in ns]

            # For the sweep_n plot, baseline line as GLOBAL mean baseline across n (simple + readable)
            base_mean_global = float(np.mean([base_by_n[n] for n in ns])) if ns else float("nan")

            fixed = {"eps": args.eps} if algo_name == "sigcGA" else {"k": args.k}
            plot_sweep(
                out_png=args.out_plot,
                x_values=xs,
                y_values=ys,
                x_label="n",
                title=f"{algo_name} sweep over n (mean evaluations), {args.fitness}, fixed {fixed}, budget={args.budget}",
                baseline_mean_eval=base_mean_global,
                budget=args.budget,
            )

        print(f"Wrote plot to {args.out_plot}")


if __name__ == "__main__":
    main()


"""""

python test.py --mode sweep_param --algo sigcGA --fitness OneMax --n 100 --budget 5000 --reps 2 --eps_grid "2, 1, 0.5, 0.05" --out_csv results_sigcga_eps_OneMax.csv --out_plot plot_sigcga_eps_OneMax.png

"""""

