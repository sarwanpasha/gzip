"""List (dataset, scheme, seed) tasks for the SLURM array. Cluster schemes are included only when
splitting was feasible for every seed (see data/splits/<ds>/cluster_id*/cluster_stats.json)."""
import argparse

from common import ensure_dir, load_config, p, read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tasks, skipped = [], []
    for ds in cfg["datasets"]:
        if "schemes" in cfg["datasets"][ds]:  # datasets with their own split schemes
            for s in cfg["datasets"][ds]["schemes"]:
                for k in range(cfg["n_seeds"]):
                    tasks.append(f"{ds} {s} {k}")
            continue
        schemes = ["random"]
        for d in sorted(p(cfg, "data", "splits", ds).glob("cluster_id*")):
            st = d / "cluster_stats.json"
            if st.exists() and read_json(st)["feasible_all_seeds"]:
                schemes.append(d.name)
            elif st.exists():
                skipped.append(f"{ds} {d.name}")
        for s in schemes:
            for k in range(cfg["n_seeds"]):
                tasks.append(f"{ds} {s} {k}")
    ensure_dir(p(cfg, "results"))
    with open(p(cfg, "results", "tasks.txt"), "w") as f:
        f.write("\n".join(tasks) + "\n")
    print(len(tasks))
    if skipped:
        with open(p(cfg, "results", "infeasible_cluster_schemes.txt"), "w") as f:
            f.write("\n".join(skipped) + "\n")


if __name__ == "__main__":
    main()
