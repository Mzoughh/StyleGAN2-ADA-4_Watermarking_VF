import json
from pathlib import Path
from typing import Any, List, Tuple
from collections import Counter

import matplotlib.pyplot as plt


def load_json_list(path: str, name: str) -> Any:
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


def jaccard_pct(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return 100.0 * inter / uni if uni else 0.0

def exact_pct(mask: List[int], ref: List[int]) -> float:
    sm, sr = set(mask), set(ref)
    inter = len(sm & sr)
    return 100.0 * inter / len(sr) if len(sr) else 0.0


def save_hist(data: List[float], title: str, xlabel: str, outpath: Path, bins: int = 30) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_hist_discrete_intersections(intersections: List[int], outpath: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(intersections, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], edgecolor="black")
    plt.title("Intersection avec ref_mask (0 à 3)")
    plt.xlabel("Nb d'indices en commun")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2, 3])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_scatter(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def align_masks_to_bitacc(masks: List[List[int]], bit_acc: List[float], threshold: float, use_ge: bool = False):
    if use_ge:
        hits = [i for i, v in enumerate(bit_acc) if v >= threshold]
        comp = ">="
    else:
        hits = [i for i, v in enumerate(bit_acc) if v > threshold]
        comp = ">"

    if len(hits) < len(masks):
        raise ValueError(
            f"Incohérence: {len(masks)} masks mais seulement {len(hits)} valeurs bit_acc {comp} {threshold}."
        )

    return [(hits[k], masks[k], bit_acc[hits[k]]) for k in range(len(masks))]


def main(bitacc_json: str, masks_json: str, threshold: float, ref_mask: List[int], outdir: str, use_ge: bool):
    out = ensure_outdir(outdir)

    bit_acc = [float(x) for x in load_json_list(bitacc_json, "bitacc.json")]
    masks_raw = load_json_list(masks_json, "masks.json")
    if not all(isinstance(m, list) for m in masks_raw):
        raise TypeError("masks.json doit être une LISTE de LISTES (ex: [[78,367,426], ...]).")
    masks = [[int(x) for x in m] for m in masks_raw]

    # 1) hist global bit_acc
    save_hist(bit_acc, "Histogramme de bit_acc (liste complète)", "bit_acc", out / "hist_bitacc_all.png", bins=30)

    # 2) hist de bit_acc > threshold (plus pertinent que recalculer 'for_masks')
    above = [v for v in bit_acc if (v >= threshold if use_ge else v > threshold)]
    save_hist(above, f"Histogramme bit_acc au-dessus du threshold={threshold}", "bit_acc", out / "hist_bitacc_above_threshold.png", bins=20)

    # 3) alignement masks <-> bit_acc
    aligned = align_masks_to_bitacc(masks, bit_acc, threshold, use_ge=use_ge)

    # jac = []
    exact_scores = []
    ba_for_masks = []
    intersections = []

    ref_set = set(ref_mask)
    rows = []  # (step_idx, mask, bit_acc, jaccard, inter)
    
    for step_idx, m, v_ba in aligned:
        e = exact_pct(m, ref_mask)
        inter = len(set(m) & ref_set)

        exact_scores.append(e)
        ba_for_masks.append(v_ba)
        intersections.append(inter)
        rows.append((step_idx, m, v_ba, e, inter))

    # 4) hist jaccard + intersection
    save_hist(exact_scores, f"Histogramme Exact(%) vs ref_mask {ref_mask}", "Exact (%)", out / "hist_exact.png", bins=10)
    save_hist_discrete_intersections(intersections, out / "hist_intersection.png")

    # 5) scatter suffisant
    save_scatter(exact_scores, ba_for_masks, "Exact (%) vs Bit accuracy", "Exact (%)", "bit_acc", out / "scatter_exact_vs_bitacc.png")

    # 6) conclusion (plus éloigné vs meilleur)
    idx_far = min(range(len(rows)), key=lambda k: rows[k][3])   # exact min
    idx_best = max(range(len(rows)), key=lambda k: rows[k][2])  # bit_acc max

    far_step, far_mask, far_ba, far_j, far_inter = rows[idx_far]
    best_step, best_mask, best_ba, best_j, best_inter = rows[idx_best]

    comp = ">=" if use_ge else ">"
    print("=== Résumé ===")
    print(f"threshold = {threshold} (bit_acc {comp} threshold) | ref_mask={ref_mask}")
    print(f"nb masks = {len(masks)} | nb hits au-dessus threshold = {len(above)}")

    print("\n--- Mask le plus éloigné (Jaccard minimal) ---")
    print(f"step={far_step} mask={far_mask} bit_acc={far_ba:.6f} jaccard={far_j:.2f}% inter={far_inter}/3")

    print("\n--- Mask meilleure bit accuracy ---")
    print(f"step={best_step} mask={best_mask} bit_acc={best_ba:.6f} jaccard={best_j:.2f}% inter={best_inter}/3")

    print(f"\nLe plus éloigné est-il aussi le meilleur ? -> {idx_far == idx_best}")

    print(f"\nPNG dans: {out.resolve()}")
    for name in [
        "hist_bitacc_all.png",
        "hist_bitacc_above_threshold.png",
        "hist_jaccard.png",
        "hist_intersection.png",
        "scatter_jaccard_vs_bitacc.png",
    ]:
        print(f" - {name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse masks vs bit_acc (JSON bruts), PNG")
    parser.add_argument("bitacc_json", help="bitacc.json = [...]")
    parser.add_argument("masks_json", help="masks.json = [[...], ...]")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--ref-mask", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--ge", action="store_true", help="Utiliser >= au lieu de >")
    args = parser.parse_args()

    main(args.bitacc_json, args.masks_json, args.threshold, args.ref_mask, args.out, args.ge)




# python analyze_all.py ./data/P_0_55/bit_acc_list_total_FP.json ./data/P_0_55/outlayer_idx_list_FP.json --threshold 0.55 --ref-mask 78 367 426 --out ./data/P_0_55/outputs
# python analyze_all.py ./data/P_0_60/bit_acc_list_total_FP.json ./data/P_0_60/outlayer_idx_list_FP.json --threshold 0.60 --ref-mask 78 367 426 --out ./data/P_0_60/outputs
# python analyze_all.py ./data/P_0_60_2POSITIONS/bit_acc_list_total_FP.json ./data/P_0_60_2POSITIONS/outlayer_idx_list_FP.json --threshold 0.60 --ref-mask 78 367 426 --out ./data/P_0_60_2POSITIONS/outputs
