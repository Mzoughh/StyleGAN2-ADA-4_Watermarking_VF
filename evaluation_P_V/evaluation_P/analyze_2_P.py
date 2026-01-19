import json
from pathlib import Path
from typing import Any, List, Tuple, Optional

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


def distance_min_max(mask: List[int], ref: List[int]) -> Optional[Tuple[float, float]]:
    """
    Calcule une distance "sans noyade" pour inter=1 ou inter=2.

    - inter == 1:
        on enlève la valeur commune.
        pour chaque valeur restante du mask, distance à la plus proche des 2 valeurs restantes du ref.
        => 2 distances -> retourne (min, max)

    - inter == 2:
        il reste 1 valeur de chaque côté.
        => retourne (d, d)

    - inter == 3:
        retourne (0,0)

    - inter == 0:
        None (non géré car ta demande ciblait inter=1 et inter=2)
    """
    sm, sr = set(mask), set(ref)
    inter = list(sm & sr)
    k = len(inter)

    if k == 3:
        return (0.0, 0.0)

    if k == 2:
        m_rest = [x for x in sm if x not in inter]  # 1 valeur
        r_rest = [x for x in sr if x not in inter]  # 1 valeur
        if len(m_rest) == 1 and len(r_rest) == 1:
            d = float(abs(m_rest[0] - r_rest[0]))
            return (d, d)
        return None

    if k == 1:
        shared = inter[0]
        m_rest = [x for x in sm if x != shared]   # 2 valeurs
        r_rest = [x for x in sr if x != shared]   # 2 valeurs
        if len(m_rest) != 2 or len(r_rest) != 2:
            return None
        # pour chaque x dans mask_rest, distance au plus proche dans ref_rest
        dists = [min(abs(x - r_rest[0]), abs(x - r_rest[1])) for x in m_rest]
        dists.sort()
        return (float(dists[0]), float(dists[1]))

    # k == 0 non géré ici
    return None


def main(bitacc_json: str, masks_json: str, threshold: float, ref_mask: List[int], outdir: str, use_ge: bool):
    out = ensure_outdir(outdir)

    bit_acc = [float(x) for x in load_json_list(bitacc_json, "bitacc.json")]
    masks_raw = load_json_list(masks_json, "masks.json")
    if not all(isinstance(m, list) for m in masks_raw):
        raise TypeError("masks.json doit être une LISTE de LISTES (ex: [[78,367,426], ...]).")
    masks = [[int(x) for x in m] for m in masks_raw]

    # 1) hist global bit_acc
    save_hist(bit_acc, "Histogramme de bit_acc (liste complète)", "bit_acc", out / "hist_bitacc_all.png", bins=30)

    # 2) hist de bit_acc > threshold
    above = [v for v in bit_acc if (v >= threshold if use_ge else v > threshold)]
    save_hist(above, f"Histogramme bit_acc au-dessus du threshold={threshold}", "bit_acc",
              out / "hist_bitacc_above_threshold.png", bins=20)

    # 3) alignement masks <-> bit_acc
    aligned = align_masks_to_bitacc(masks, bit_acc, threshold, use_ge=use_ge)

    exact_scores = []
    ba_for_masks = []
    intersections = []

    # pour analyse distances inter=1 et inter=2
    inter1_distmin, inter1_distmax, inter1_ba = [], [], []
    inter2_distmin, inter2_distmax, inter2_ba = [], [], []

    ref_set = set(ref_mask)
    # rows: (step_idx, mask, bit_acc, exact, inter)
    rows = []

    for step_idx, m, v_ba in aligned:
        e = exact_pct(m, ref_mask)
        inter = len(set(m) & ref_set)

        exact_scores.append(e)
        ba_for_masks.append(v_ba)
        intersections.append(inter)
        rows.append((step_idx, m, v_ba, e, inter))

        # distances
        dm = distance_min_max(m, ref_mask)
        if dm is not None:
            dmin, dmax = dm
            if inter == 1:
                inter1_distmin.append(dmin)
                inter1_distmax.append(dmax)
                inter1_ba.append(v_ba)
            elif inter == 2:
                inter2_distmin.append(dmin)
                inter2_distmax.append(dmax)
                inter2_ba.append(v_ba)

    # 4) hist exact + intersection
    save_hist(exact_scores, f"Histogramme Exact(%) vs ref_mask {ref_mask}", "Exact (%)",
              out / "hist_exact.png", bins=10)
    save_hist_discrete_intersections(intersections, out / "hist_intersection.png")

    # 5) scatter principal
    save_scatter(exact_scores, ba_for_masks, "Exact (%) vs Bit accuracy", "Exact (%)", "bit_acc",
                 out / "scatter_exact_vs_bitacc.png")

    # 5bis) scatters distances (inter=1)
    if inter1_distmin:
        save_scatter(
            inter1_distmin, inter1_ba,
            "Intersection=1 : dist_min vers target restante vs bit_acc",
            "dist_min (plus proche des 2 valeurs restantes)",
            "bit_acc",
            out / "scatter_inter1_distmin_vs_bitacc.png"
        )
    if inter1_distmax:
        save_scatter(
            inter1_distmax, inter1_ba,
            "Intersection=1 : dist_max vers target restante vs bit_acc",
            "dist_max (plus loin des 2 valeurs restantes)",
            "bit_acc",
            out / "scatter_inter1_distmax_vs_bitacc.png"
        )

    # 5ter) scatters distances (inter=2)
    if inter2_distmin:
        save_scatter(
            inter2_distmin, inter2_ba,
            "Intersection=2 : distance (valeur restante) vs bit_acc",
            "distance (|mask_rest - target_rest|)",
            "bit_acc",
            out / "scatter_inter2_dist_vs_bitacc.png"
        )

    # 6) conclusion (exact min vs best bitacc)
    idx_far = min(range(len(rows)), key=lambda k: rows[k][3])   # exact min
    idx_best = max(range(len(rows)), key=lambda k: rows[k][2])  # bit_acc max

    far_step, far_mask, far_ba, far_e, far_inter = rows[idx_far]
    best_step, best_mask, best_ba, best_e, best_inter = rows[idx_best]

    comp = ">=" if use_ge else ">"
    print("=== Résumé ===")
    print(f"threshold = {threshold} (bit_acc {comp} threshold) | ref_mask={ref_mask}")
    print(f"nb masks = {len(masks)} | nb hits au-dessus threshold = {len(above)}")

    print("\n--- Mask Exact minimal ---")
    print(f"step={far_step} mask={far_mask} bit_acc={far_ba:.6f} exact={far_e:.1f}% inter={far_inter}/3")

    print("\n--- Mask meilleure bit accuracy ---")
    print(f"step={best_step} mask={best_mask} bit_acc={best_ba:.6f} exact={best_e:.1f}% inter={best_inter}/3")

    print(f"\nExact minimal est-il aussi le meilleur ? -> {idx_far == idx_best}")

    print(f"\nPNG dans: {out.resolve()}")
    for name in [
        "hist_bitacc_all.png",
        "hist_bitacc_above_threshold.png",
        "hist_exact.png",
        "hist_intersection.png",
        "scatter_exact_vs_bitacc.png",
        "scatter_inter1_distmin_vs_bitacc.png",
        "scatter_inter1_distmax_vs_bitacc.png",
        "scatter_inter2_dist_vs_bitacc.png",
    ]:
        p = out / name
        if p.exists():
            print(f" - {name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse masks vs bit_acc (JSON bruts), PNG + distances min/max")
    parser.add_argument("bitacc_json", help="bitacc.json = [...]")
    parser.add_argument("masks_json", help="masks.json = [[...], ...]")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--ref-mask", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--ge", action="store_true", help="Utiliser >= au lieu de >")
    args = parser.parse_args()

    main(args.bitacc_json, args.masks_json, args.threshold, args.ref_mask, args.out, args.ge)

# python analyze_2.py ./data/P_0_55/bit_acc_list_total_FP.json ./data/P_0_55/outlayer_idx_list_FP.json --threshold 0.55 --ref-mask 78 367 426 --out ./data/P_0_55/outputs
# python analyze_2.py ./data/P_0_60/bit_acc_list_total_FP.json ./data/P_0_60/outlayer_idx_list_FP.json --threshold 0.60 --ref-mask 78 367 426 --out ./data/P_0_60/outputs
# python analyze_2.py ./data/P_0_60_2POSITIONS/bit_acc_list_total_FP.json ./data/P_0_60_2POSITIONS/outlayer_idx_list_FP.json --threshold 0.60 --ref-mask 78 367 426 --out ./data/P_0_60_2POSITIONS/outputs
