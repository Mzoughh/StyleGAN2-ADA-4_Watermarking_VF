import os
import re
import argparse
from PIL import Image


def attack_sort_key(severity: str, attack_type: str):
    """
    Clé de tri pour les SEVERITÉS d'une même attaque :
      - 'none' en premier
      - pour 'quant'/'quantization' : plus le nombre est grand, moins l'attaque
        est sévère -> on trie par nombres décroissants pour aller du moins
        impactant au plus impactant (8, 4, 2, ...)
      - pour les autres (ex: pruning) : on trie par nombres croissants (5, 10, 50)
      - si pas de nombre, tri lexicographique.
    """
    if severity in ("none", "", None):
        return (0, (), severity or "")

    nums = re.findall(r"\d+\.?\d*", severity)
    if nums:
        numeric_tuple = tuple(float(x) for x in nums)

        # Cas quantization : on inverse le signe pour trier dans l'ordre
        # "impact croissant" (ex: 8 -> 4 -> 2)
        if attack_type.startswith("quant"):
            inv = tuple(-x for x in numeric_tuple)
            return (1, inv, severity)
        else:
            # pruning, jpeg, etc. : tri croissant normal
            return (1, numeric_tuple, severity)

    # Pas de nombre -> à la fin, trié par nom
    return (1, (float("inf"),), severity)


def parse_attack_from_filename(fname: str):
    """
    À partir d'un nom de fichier du type :
        fakes_after_pruning_5.png
        fakes_after_pruning_50.png
        fakes_after_quantization_2_8.png
        fakes_after_none.png

    renvoie:
        attack_type   : 'pruning', 'quantization', 'none', ...
        severity_name : '5', '50', '2_8', 'none', ...
    """
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)

    prefix = "fakes_after_"
    if not stem.startswith(prefix):
        # fallback brut
        return stem, ""

    rest = stem[len(prefix):]  # ex: 'pruning_5', 'quantization_2_8', 'none'

    if rest == "" or rest == "none":
        return "none", "none"

    parts = rest.split("_", 1)
    if len(parts) == 1:
        return parts[0], ""  # ex: 'pruning'
    attack_type, severity = parts[0], parts[1]  # ex: 'pruning', '5'
    return attack_type, severity


def extract_first_face(img: Image.Image) -> Image.Image:
    """
    Extrait le premier visage (tuile en haut à gauche) d'une grille 4x4.
    """
    w, h = img.size
    tile_w = w // 4
    tile_h = h // 4
    return img.crop((0, 0, tile_w, tile_h))


def find_before_attack_image(directory: str):
    """
    Cherche un fichier 'fake_before_attack*.png' ou 'fakes_before_attack*.png'
    pour servir de référence 'avant attaque'.
    """
    candidates = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".png")
        and (f.startswith("fake_before_attack") or f.startswith("fakes_before_attack"))
    ]
    if not candidates:
        return None
    # On prend le premier trouvé
    return os.path.join(directory, sorted(candidates)[0])


def build_composites_by_attack_type(directory: str, pattern_prefix: str = "fakes_after_"):
    """
    Pour chaque type d'attaque (pruning, quantization, ...), construit une image
    où l'on aligne horizontalement le premier visage de chaque grille,
    trié par sévérité croissante (au sens de l'impact).
    Au début de chaque ligne, on ajoute le visage issu de fake_before_attack.
    """
    files = [
        f for f in os.listdir(directory)
        if f.startswith(pattern_prefix) and f.lower().endswith(".png")
    ]

    if not files:
        print(f"Aucun fichier '{pattern_prefix}*.png' trouvé dans {directory}")
        return

    # Image avant attaque (référence)
    before_attack_path = find_before_attack_image(directory)
    if before_attack_path:
        print(f"Image 'before attack' utilisée : {os.path.basename(before_attack_path)}")
    else:
        print("Aucune image 'fake_before_attack*.png' trouvée, pas de référence avant attaque.")

    # Regrouper par type d'attaque
    attacks_by_type = {}  # attack_type -> list of (severity_name, filepath)
    baseline_after_none = None

    for f in files:
        attack_type, severity = parse_attack_from_filename(f)
        full_path = os.path.join(directory, f)

        if attack_type == "none":
            baseline_after_none = full_path
            continue

        attacks_by_type.setdefault(attack_type, []).append((severity, full_path))

    if not attacks_by_type and not before_attack_path and not baseline_after_none:
        print("Uniquement une baseline 'none' trouvée, rien à comparer.")
        return

    # Déterminer la taille d'une tuile : priorité à l'image before-attack,
    # sinon baseline none, sinon une attaque quelconque.
    sample_img_path = before_attack_path or baseline_after_none
    if sample_img_path is None:
        any_type = next(iter(attacks_by_type.keys()))
        sample_img_path = attacks_by_type[any_type][0][1]

    with Image.open(sample_img_path) as im0:
        example_face = extract_first_face(im0)
        tile_w, tile_h = example_face.size

    # Pour chaque type d'attaque, construire une image
    for attack_type, entries in attacks_by_type.items():
        # entries : liste de (severity_name, path)
        # tri par ordre "impact croissant" adapté au type
        entries.sort(key=lambda x: attack_sort_key(x[0], attack_type))

        faces = []
        labels = []

        # 1) Visage avant attaque (si dispo)
        if before_attack_path is not None:
            with Image.open(before_attack_path) as im_before:
                faces.append(extract_first_face(im_before))
                labels.append("before")

        # 2) Baseline après 'none' (si tu veux aussi la voir - on la met après 'before')
        if baseline_after_none is not None:
            with Image.open(baseline_after_none) as im_none:
                faces.append(extract_first_face(im_none))
                labels.append("none")

        print(f"\nType d'attaque : {attack_type}")
        print("Ordre des sévérités :")

        # 3) Toutes les sévérités de ce type
        for severity, path in entries:
            with Image.open(path) as im:
                faces.append(extract_first_face(im))
            labels.append(severity)
            print(f"  - {severity}  <- {os.path.basename(path)}")

        if not faces:
            continue

        n = len(faces)
        composite = Image.new("RGB", (tile_w * n, tile_h))

        for i, face in enumerate(faces):
            composite.paste(face, (i * tile_w, 0))

        out_name = f"first_faces_{attack_type}.png"
        out_path = os.path.join(directory, out_name)
        composite.save(out_path)
        print(f"Image composée sauvegardée pour '{attack_type}' dans : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrait le premier visage de chaque grille fakes_after_*.png et, "
            "pour chaque type d'attaque (pruning, quantization, ...), génère "
            "une image montrant la dégradation en fonction des paramètres. "
            "Ajoute au début le visage du fichier fake_before_attack*.png."
        )
    )
    parser.add_argument(
        "directory",
        help="Dossier contenant les fichiers fakes_after_*.png et fake_before_attack*.png"
    )
    args = parser.parse_args()

    build_composites_by_attack_type(args.directory)


if __name__ == "__main__":
    main()
