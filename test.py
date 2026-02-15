# experiments.py
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Iterable, Optional

import numpy as np

from FitnessFunctions import OneMax  # add LeadingOnes, Jump as you implement them
from TerminationCriterion import TerminationCriterion
from cGA import cGA
from sigcGA import sigcGA
from onePlusOneEA import onePlusOneEA


# ---------- Configuration objects ----------

@dataclass(frozen=True)
class AlgoConfig:
    name: str
    # params holds algo-specific parameters (e.g., k for cGA, eps for sigcGA)
    params: Dict[str, Any]

@dataclass(frozen=True)
class ExperimentConfig:
    fitness_name: str
    n: int
    budget: int
    algo: AlgoConfig

@dataclass
class RunResult:
    seed: int
    fitness_name: str
    n: int
    budget: int
    algo_name: str
    algo_params: str  # stored as string for CSV convenience
    iterations: int
    evaluations: int
    # optional: you can store summary stats of p if you want (avoid huge vectors)
    p_mean: float
    p_min: float
    p_max: float


# ---------- Factories (fresh state each run) ----------

def make_fitness(name: str):
    if name == "OneMax":
        return OneMax()
    raise ValueError(f"Unknown fitness: {name}")

def make_termination(fitness, n: int, budget: int):
    return TerminationCriterion(fitness=fitness, n=n, budget=budget)


# ---------- Runner for a single run ----------

def run_once(cfg: ExperimentConfig, seed: int) -> RunResult:
    # Create RNGs for THIS repetition
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Fresh fitness + termination each run (avoid call_count contamination)
    fitness = make_fitness(cfg.fitness_name)
    termination = make_termination(fitness, cfg.n, cfg.budget)

    algo = cfg.algo.name
    params = cfg.algo.params

    if algo == "onePlusOneEA":
        p, t = onePlusOneEA(cfg.n, fitness, termination, py_rng, np_rng)

    elif algo == "cGA":
        k = params["k"]
        p, t = cGA(cfg.n, k, fitness, termination, py_rng, np_rng)

    elif algo == "sigcGA":
        eps = params["eps"]
        p, t = sigcGA(cfg.n, eps, fitness, termination, py_rng, np_rng)

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    vec = np.array(p.vector, dtype=float)
    return RunResult(
        seed=seed,
        fitness_name=cfg.fitness_name,
        n=cfg.n,
        budget=cfg.budget,
        algo_name=algo,
        algo_params=str(params),
        iterations=int(t),
        evaluations=int(fitness.call_count),
        p_mean=float(vec.mean()),
        p_min=float(vec.min()),
        p_max=float(vec.max()),
    )


# ---------- Runner for many runs / configs ----------

def iter_seeds(base_seed: int, count: int) -> Iterable[int]:
    # deterministic seed schedule: base_seed, base_seed+1, ...
    for i in range(count):
        yield base_seed + i

def run_repetitions(
    cfg: ExperimentConfig,
    repetitions: int,
    base_seed: int,
) -> list[RunResult]:
    return [run_once(cfg, seed) for seed in iter_seeds(base_seed, repetitions)]

def run_grid(
    configs: list[ExperimentConfig],
    repetitions: int,
    base_seed: int,
) -> list[RunResult]:
    all_results: list[RunResult] = []
    # give each config its own seed block so reordering configs doesn't change results
    seed_stride = 10_000

    for idx, cfg in enumerate(configs):
        cfg_base = base_seed + idx * seed_stride
        all_results.extend(run_repetitions(cfg, repetitions, cfg_base))
    return all_results


# ---------- Output ----------

def write_csv(path: str, results: list[RunResult]) -> None:
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


# ---------- Example Task-6 style usage ----------

def main():
    repetitions = 30
    base_seed = 42

    # Example grid (extend this massively for Task 6)
    configs = [
        ExperimentConfig(
            fitness_name="OneMax",
            n=100,
            budget=100_000,
            algo=AlgoConfig(name="onePlusOneEA", params={}),
        ),
        ExperimentConfig(
            fitness_name="OneMax",
            n=100,
            budget=100_000,
            algo=AlgoConfig(name="cGA", params={"k": 50}),
        ),
        ExperimentConfig(
            fitness_name="OneMax",
            n=100,
            budget=100_000,
            algo=AlgoConfig(name="sigcGA", params={"eps": 0.05}),
        ),
    ]

    results = run_grid(configs, repetitions=repetitions, base_seed=base_seed)
    write_csv("results_task6.csv", results)

    # Tiny console summary
    print(f"Wrote {len(results)} rows to results_task6.csv")
    # show a quick mean evals per config
    by_key: Dict[str, list[int]] = {}
    for r in results:
        key = f"{r.fitness_name}|n={r.n}|{r.algo_name}|{r.algo_params}"
        by_key.setdefault(key, []).append(r.evaluations)

    for key, vals in by_key.items():
        print(f"{key}: mean_evals={sum(vals)/len(vals):.1f} over {len(vals)} runs")


if __name__ == "__main__":
    main()
