import os
import json
import argparse
import csv
import re
import matplotlib.pyplot as plt


# ---------- Config : métriques à ignorer dans les tableaux ----------
IGNORED_METRICS = {"uchida_hamming_dist"}


# ---------- Helpers pour les labels ----------

def latex_escape(s: str) -> str:
    """
    Échappe les caractères spéciaux LaTeX dans une chaîne.
    À NE PAS utiliser sur des labels déjà en syntaxe LaTeX.
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for orig, repl in replacements.items():
        s = s.replace(orig, repl)
    return s


def metric_latex_label(metric: str) -> str:
    """
    Renvoie un label joli pour le papier, en LaTeX.
    """
    mapping = {
        "fid50k_full": r"FID$_{50k}$ ($\downarrow$)",
        "uchida_bit_acc": r"Bit acc. ($\uparrow$)",
        # tu peux ajouter d'autres mappings ici
    }
    if metric in mapping:
        return mapping[metric]
    return latex_escape(metric)


def metric_pretty_label(metric: str) -> str:
    """
    Label "joli" pour PNG (matplotlib).
    """
    return metric_latex_label(metric)


# ---------- Parsing du nom de fichier ----------

def parse_attack_from_filename(fname: str):
    """
    À partir d'un nom de fichier du type :
        metric-fid50k_full.json
        metric-fid50k_full-none.json
        metric-fid50k_full-pruning-50.json
        metric-uchida_extraction-pruning-50.json

    renvoie:
        attack_type : 'none', 'pruning', 'jpeg', ...
        attack_name : 'none', 'pruning-50', 'jpeg-10', ...
    """
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)

    if not stem.startswith("metric-"):
        return "unknown", "unknown"

    rest = stem[len("metric-"):]  # ex: "fid50k_full-pruning-50"
    parts = rest.split("-")

    # Pas de suffixe → baseline
    if len(parts) == 1:
        return "none", "none"

    # Suffixe explicite "none" → baseline
    if parts[1] == "none":
        return "none", "none"

    # Cas général : metric_family-attack_type-params...
    attack_type = parts[1]
    attack_name = "-".join(parts[1:])  # "pruning-50", "jpeg-10", etc.
    return attack_type, attack_name


# ---------- Chargement des métriques ----------

def load_evaluation_metrics_per_type(directory: str):
    """
    Lit tous les fichiers metric-* dans le dossier.
    Chaque fichier contient UNE SEULE structure JSON.

    Renvoie:
        tables_by_type   : dict[attack_type][attack_name][metric_key] = valeur
        metrics_by_type  : dict[attack_type] -> set(noms de métriques)
    """
    files = [
        f for f in os.listdir(directory)
        if f.startswith("metric-") and os.path.isfile(os.path.join(directory, f))
    ]

    if not files:
        print(f"Aucun fichier 'metric-*' trouvé dans {directory}")
        return {}, {}

    tables_by_type = {}      # attack_type -> { attack_name -> { metric_key: value } }
    metrics_by_type = {}     # attack_type -> set(metric_key)

    for fname in sorted(files):
        path = os.path.join(directory, fname)

        # Lecture du JSON complet
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON invalide dans {fname} : {e}")
            continue

        results = data.get("results", {})
        if not isinstance(results, dict) or not results:
            print(f"[WARN] Champ 'results' vide ou invalide dans : {fname}")
            continue

        attack_type, attack_name = parse_attack_from_filename(fname)

        # Structures pour ce type d'attaque
        type_table = tables_by_type.setdefault(attack_type, {})
        type_metrics_set = metrics_by_type.setdefault(attack_type, set())

        attack_metrics = type_table.setdefault(attack_name, {})

        # Ajoute toutes les métriques
        for metric_key, value in results.items():
            type_metrics_set.add(metric_key)
            attack_metrics[metric_key] = value

    return tables_by_type, metrics_by_type


# ---------- Sauvegarde CSV / LaTeX / PNG ----------

def save_table_csv(attacks, metrics, table_for_type, output_path: str):
    # Première cellule d'en-tête vide, puis métriques
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + metrics)
        for attack in attacks:
            row = [attack]
            for metric in metrics:
                val = table_for_type.get(attack, {}).get(metric, "")
                row.append(val)
            writer.writerow(row)
    print(f"Tableau CSV sauvegardé dans : {output_path}")


def save_table_latex(attacks, metrics, table_for_type, output_path: str, attack_type: str):
    with open(output_path, "w") as f:
        # %% pour avoir un % littéral
        f.write("%% Auto-generated evaluation table (attack type: %s)\n" % attack_type)
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")

        # 1ère colonne = labels de lignes (attaques), ensuite les métriques
        col_spec = "l" + "c" * len(metrics)
        f.write("\\begin{tabular}{%s}\n" % col_spec)
        f.write("\\hline\n")

        # Première cellule vide (colonne des attaques), puis colonnes de métriques
        header_cells = [""] + [metric_latex_label(m) for m in metrics]
        f.write(" & ".join(header_cells) + " \\\\\n")
        f.write("\\hline\n")

        for attack in attacks:
            row_cells = [latex_escape(attack)]
            for metric in metrics:
                val = table_for_type.get(attack, {}).get(metric, "")
                if isinstance(val, float):
                    row_cells.append(f"{val:.4f}")
                else:
                    row_cells.append(latex_escape(str(val)))
            f.write(" & ".join(row_cells) + " \\\\\n")

        f.write("\\hline\n")
        f.write(
            "\\caption{Évaluation des métriques pour l'attaque \\texttt{%s}.}\n"
            % latex_escape(attack_type)
        )
        f.write("\\label{tab:%s_metrics}\n" % attack_type.replace(" ", "_"))
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Tableau LaTeX sauvegardé dans : {output_path}")


def save_table_png(attacks, metrics, table_for_type, output_path: str, attack_type: str):
    """
    Sauvegarde le tableau sous forme d'image PNG, style sobre.
    Les attaques sont des labels de lignes, les colonnes sont uniquement des métriques.
    """
    col_labels = [metric_pretty_label(m) for m in metrics]
    cell_text = []

    for attack in attacks:
        row = []
        for metric in metrics:
            val = table_for_type.get(attack, {}).get(metric, "")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        cell_text.append(row)

    # Style "latex-friendly"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 9

    fig_width = max(6, 1.2 * len(metrics))
    fig_height = max(2.5, 0.5 * (len(attacks) + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table_obj = ax.table(
        cellText=cell_text,
        rowLabels=attacks,          # noms d'attaques à gauche
        colLabels=col_labels,       # uniquement les métriques
        loc="center"
    )

    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(9)
    table_obj.scale(1.2, 1.2)

    ax.set_title(f"Attack type: {attack_type}", pad=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Tableau PNG sauvegardé dans : {output_path}")


# ---------- Utilitaires ----------

def attack_sort_key(name: str):
    """
    Clé de tri pour les attaques :
      - 'none' en premier
      - ensuite tri croissant sur l'ensemble des nombres trouvés dans le nom
        (ex: pruning-5, pruning-10, pruning-50 ; quant-2-8, quant-4-4, etc.)
      - si pas de nombre, tri lexicographique.
    """
    if name == "none":
        return (0, (), name)

    nums = re.findall(r"\d+\.?\d*", name)
    if nums:
        numeric_tuple = tuple(float(x) for x in nums)
        return (1, numeric_tuple, name)

    # Pas de nombre dans le nom -> on met à la fin, trié par nom
    return (1, (float("inf"),), name)


def sort_attacks(attacks):
    return sorted(attacks, key=attack_sort_key)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construit un tableau (attaques x métriques) par type d'attaque "
            "à partir des fichiers metric-* d'un dossier d'évaluation, et "
            "le sauvegarde en CSV + LaTeX + PNG (style paper-friendly)."
        )
    )
    parser.add_argument(
        "directory",
        help="Dossier contenant les fichiers metric-*.json/jsonl d'évaluation"
    )
    args = parser.parse_args()

    tables_by_type, metrics_by_type = load_evaluation_metrics_per_type(args.directory)
    if not tables_by_type:
        print("Rien à tracer / sauvegarder.")
        return

    # ---- Baseline 'none' (réseau non attaqué) ----
    baseline_metrics = None
    if "none" in tables_by_type:
        baseline_metrics = tables_by_type["none"].get("none", None)

    if baseline_metrics:
        # Recopie la ligne 'none' dans toutes les autres tables
        for attack_type, table_for_type in tables_by_type.items():
            if attack_type == "none":
                continue
            table_for_type.setdefault("none", baseline_metrics.copy())
            metrics_by_type.setdefault(attack_type, set()).update(baseline_metrics.keys())

    # ---- Génération des tableaux par type d'attaque ----
    for attack_type, table_for_type in tables_by_type.items():
        attacks = sort_attacks(list(table_for_type.keys()))

        # Métriques à afficher = toutes sauf celles ignorées
        all_metrics = sorted(metrics_by_type.get(attack_type, []))
        metrics = [m for m in all_metrics if m not in IGNORED_METRICS]

        if not attacks or not metrics:
            continue

        csv_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.csv")
        tex_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.tex")
        png_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.png")

        save_table_csv(attacks, metrics, table_for_type, csv_path)
        save_table_latex(attacks, metrics, table_for_type, tex_path, attack_type)
        save_table_png(attacks, metrics, table_for_type, png_path, attack_type)


if __name__ == "__main__":
    main()
