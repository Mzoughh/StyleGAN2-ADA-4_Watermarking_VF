import json
from pathlib import Path
from typing import Any, List, Dict, Tuple
import math
import csv

import matplotlib.pyplot as plt


def load_json_list(path: str, name: str) -> List[Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{name}: le JSON doit contenir une LISTE à la racine (pas de clé).")
    return data


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def ci95(xs: List[float]) -> float:
    """Half-width of 95% CI using normal approx: 1.96 * std/sqrt(n)"""
    n = len(xs)
    if n <= 1:
        return 0.0
    _, s = mean_std(xs)
    return 1.96 * s / math.sqrt(n)


def save_scatter(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_errorbar(x: List[float], y: List[float], yerr: List[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_boxplot(groups: List[List[float]], labels: List[str], title: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(bitacc_json: str, cvalue_json: str, outdir: str = "outputs") -> None:
    out = ensure_outdir(outdir)

    bit_raw = load_json_list(bitacc_json, "bit_acc_list_total_FP.json")
    c_raw = load_json_list(cvalue_json, "c_value_list_total_FP.json")

    if len(bit_raw) != len(c_raw):
        raise ValueError(f"Longueurs différentes: bit_acc={len(bit_raw)} vs c_value={len(c_raw)}")

    # cast
    bit_acc = [float(x) for x in bit_raw]
    c_vals = [float(x) for x in c_raw]

    # 1) Scatter brut
    save_scatter(
        c_vals, bit_acc,
        "Bit accuracy vs c_value (scatter brut)",
        "c_value ajouté",
        "bit_acc",
        out / "scatter_bitacc_vs_cvalue.png"
    )

    # 2) Regroupement par c_value
    groups: Dict[float, List[float]] = {}
    for c, b in zip(c_vals, bit_acc):
        groups.setdefault(c, []).append(b)

    # tri par c_value
    c_sorted = sorted(groups.keys())
    means, stds, cis, ns = [], [], [], []
    for c in c_sorted:
        xs = groups[c]
        m, s = mean_std(xs)
        means.append(m)
        stds.append(s)
        cis.append(ci95(xs))
        ns.append(len(xs))

    # 3) Plot mean ± std
    save_errorbar(
        c_sorted, means, stds,
        "Bit accuracy moyenne par c_value (± std)",
        "c_value ajouté",
        "bit_acc (mean)",
        out / "mean_std_bitacc_by_cvalue.png"
    )

    # 4) Plot mean ± CI95
    save_errorbar(
        c_sorted, means, cis,
        "Bit accuracy moyenne par c_value (IC 95%)",
        "c_value ajouté",
        "bit_acc (mean)",
        out / "mean_ci95_bitacc_by_cvalue.png"
    )

    # 5) Boxplot par c_value (distribution)
    labels = [str(int(c)) if c.is_integer() else str(c) for c in c_sorted]
    grouped_values = [groups[c] for c in c_sorted]
    save_boxplot(
        grouped_values, labels,
        "Distribution de bit_acc par c_value (boxplot)",
        "bit_acc",
        out / "boxplot_bitacc_by_cvalue.png"
    )

    # 6) CSV résumé
    csv_path = out / "summary_by_cvalue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["c_value", "n", "mean_bitacc", "std_bitacc", "ci95_halfwidth"])
        for c, n, m, s, ci in zip(c_sorted, ns, means, stds, cis):
            w.writerow([c, n, m, s, ci])

    # 7) Interprétation imprimée rapide
    best_idx = max(range(len(means)), key=lambda i: means[i])
    worst_idx = min(range(len(means)), key=lambda i: means[i])

    print("=== Analyse c_value -> bit_acc ===")
    print(f"Total points: {len(bit_acc)} | c_values uniques: {len(c_sorted)}")
    print("Par c_value:")
    for c, n, m, s in zip(c_sorted, ns, means, stds):
        print(f"  c={c:>6}: n={n:>4} mean={m:.6f} std={s:.6f}")
    print()
    print(f"Meilleur c_value (mean): c={c_sorted[best_idx]} -> {means[best_idx]:.6f}")
    print(f"Pire   c_value (mean): c={c_sorted[worst_idx]} -> {means[worst_idx]:.6f}")

    print("\nFichiers générés:")
    for name in [
        "scatter_bitacc_vs_cvalue.png",
        "mean_std_bitacc_by_cvalue.png",
        "mean_ci95_bitacc_by_cvalue.png",
        "boxplot_bitacc_by_cvalue.png",
        "summary_by_cvalue.csv",
    ]:
        p = out / name
        print(f" - {p.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse bit_acc en fonction de c_value (JSON listes brutes)")
    parser.add_argument("bitacc_json", help="bit_acc_list_total_FP.json (liste)")
    parser.add_argument("cvalue_json", help="c_value_list_total_FP.json (liste)")
    parser.add_argument("--out", default="outputs", help="Dossier de sortie")
    args = parser.parse_args()

    main(args.bitacc_json, args.cvalue_json, outdir=args.out)

# python analyze_V.py ./data/bit_acc_list_total_FP.json ./data/c_value_list_total_FP.json --out ./data/outputs_c_value