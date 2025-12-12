import os
import json
import argparse
import matplotlib.pyplot as plt
import re
from itertools import cycle

IGNORED_METRICS = {"uchida_hamming_dist"}

# Palette de couleurs agréable (matplotlib "tab" colors)
COLOR_CYCLE = cycle([
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
])


def parse_step_from_snapshot(snapshot_pkl: str, fallback: int) -> int:
    """
    Extrait le numéro de step à partir de 'network-snapshot-000010.pkl'.
    Si impossible, renvoie fallback.
    """
    if not snapshot_pkl:
        return fallback

    m = re.search(r"(\d+)(?=\.pkl$)", snapshot_pkl)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return fallback


def load_all_metrics(directory: str):
    """
    Parcourt tous les fichiers commençant par 'metric-' dans le dossier,
    lit chaque ligne comme un JSON, et agrège toutes les métriques trouvées
    dans un dict : { metric_name: [(step, value), ...], ... }.
    Ignore les métriques listées dans IGNORED_METRICS.
    """
    metric_files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("metric-") and os.path.isfile(os.path.join(directory, f))
    )

    if not metric_files:
        print(f"Aucun fichier 'metric-*' trouvé dans {directory}")
        return {}

    all_metrics = {}  # metric_name -> list of (step, value)

    for fname in metric_files:
        path = os.path.join(directory, fname)
        print(f"Lecture du fichier de métriques : {path}")

        with open(path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Ligne ignorée dans {fname} (JSON invalide) : {e}")
                    continue

                snapshot_pkl = data.get("snapshot_pkl", "")
                step = parse_step_from_snapshot(snapshot_pkl, idx)

                results = data.get("results", {})
                for metric_name, value in results.items():
                    if metric_name in IGNORED_METRICS:
                        continue
                    all_metrics.setdefault(metric_name, []).append((step, value))

    return all_metrics


def _build_title(metric_names):
    """Construit un titre plus représentatif en fonction des métriques tracées."""
    metric_names = list(metric_names)
    n = len(metric_names)
    if n == 1:
        return f"Évolution de {metric_names[0]} pendant l'entraînement"
    elif n == 2:
        return f"Évolution de {metric_names[0]} et {metric_names[1]} pendant l'entraînement"
    else:
        return f"Évolution de {n} métriques pendant l'entraînement"


def plot_all_metrics(all_metrics: dict, directory: str, normalize: bool = True):
    """
    Trace les métriques trouvées.

    - Si 0 métrique : ne fait rien.
    - Si 1 métrique  : un seul axe Y, valeurs brutes.
    - Si 2 métriques : double axe Y (twinx), valeurs brutes, couleurs différentes.
    - Si >2 métriques:
        * si normalize=True : normalisation par métrique dans [0, 1] sur un seul axe.
        * si normalize=False : valeurs brutes sur le même axe (attention aux échelles).
    """
    # Filtre final (au cas où)
    metrics_items = [(name, points) for name, points in all_metrics.items()
                     if name not in IGNORED_METRICS]

    if not metrics_items:
        print("Aucune métrique à tracer.")
        return

    # Tri par nom pour stabilité
    metrics_items.sort(key=lambda x: x[0])
    metric_names = [m for m, _ in metrics_items]
    n_metrics = len(metrics_items)

    # --- Cas 1 : une seule métrique ---
    if n_metrics == 1:
        metric_name, points = metrics_items[0]
        points = sorted(points, key=lambda x: x[0])
        steps = [p[0] for p in points]
        values = [p[1] for p in points]

        color = next(COLOR_CYCLE)

        plt.figure()
        plt.plot(steps, values, marker="o", label=metric_name, color=color)
        plt.xlabel("Training step")
        plt.ylabel(metric_name)
        # plt.title(_build_title([metric_name]))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # --- Cas 2 : double axe Y ---
    elif n_metrics == 2:
        (m1, pts1), (m2, pts2) = metrics_items

        pts1 = sorted(pts1, key=lambda x: x[0])
        pts2 = sorted(pts2, key=lambda x: x[0])
        steps1 = [p[0] for p in pts1]
        values1 = [p[1] for p in pts1]
        steps2 = [p[0] for p in pts2]
        values2 = [p[1] for p in pts2]

        c1 = next(COLOR_CYCLE)
        c2 = next(COLOR_CYCLE)

        fig, ax1 = plt.subplots()

        l1, = ax1.plot(steps1, values1, marker="o", label=m1, color=c1)
        ax1.set_xlabel("Training step")
        ax1.set_ylabel(m1, color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)
        ax1.grid(True)

        ax2 = ax1.twinx()
        l2, = ax2.plot(steps2, values2, marker="s", linestyle="--", label=m2, color=c2)
        ax2.set_ylabel(m2, color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)

        # Légende combinée
        lines = [l1, l2]
        labels = [m1, m2]
        ax1.legend(lines, labels, loc="best")

        # plt.title(_build_title([m1, m2]))
        fig.tight_layout()

    # --- Cas > 2 métriques ---
    else:
        plt.figure()

        for metric_name, points in metrics_items:
            points = sorted(points, key=lambda x: x[0])
            steps = [p[0] for p in points]
            values = [p[1] for p in points]

            if normalize:
                vmin = min(values)
                vmax = max(values)
                if vmax > vmin:
                    norm_values = [(v - vmin) / (vmax - vmin) for v in values]
                else:
                    norm_values = [0.5] * len(values)  # Courbe constante
                plot_values = norm_values
            else:
                plot_values = values

            color = next(COLOR_CYCLE)
            plt.plot(steps, plot_values, marker="o", label=metric_name, color=color)

        plt.xlabel("Training step")
        plt.ylabel("Valeur normalisée de la métrique" if normalize else "Metric value")
        # plt.title(_build_title(metric_names) +
        #           (" (normalisées)" if normalize else ""))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    output_path = os.path.join(directory, "metrics_training_curves.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Figure sauvegardée dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace les métriques trouvées dans les fichiers 'metric-*'."
    )
    parser.add_argument(
        "directory",
        help="Dossier contenant les fichiers metric-* (ex: metric-fid50k_full.jsonl, metric-uchida_extraction.jsonl, ...)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Désactiver la normalisation quand il y a plus de 2 métriques."
    )
    args = parser.parse_args()

    all_metrics = load_all_metrics(args.directory)
    plot_all_metrics(all_metrics, args.directory, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
