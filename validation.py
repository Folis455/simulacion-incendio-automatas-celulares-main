import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_mask(path: str) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    mask = img.astype(float)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask >= 0.5)


def compute_iou(sim_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(sim_mask, gt_mask).sum()
    union = np.logical_or(sim_mask, gt_mask).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def compute_relative_area_error(sim_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    a_sim = sim_mask.sum()
    a_gt = gt_mask.sum()
    if a_gt == 0:
        return 0.0 if a_sim == 0 else np.inf
    return abs(a_sim - a_gt) / float(a_gt) * 100.0


def build_diff_overlay(sim_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    h, w = sim_mask.shape
    overlay = np.zeros((h, w, 3), dtype=float)
    tp = np.logical_and(sim_mask, gt_mask)
    fp = np.logical_and(sim_mask, np.logical_not(gt_mask))
    fn = np.logical_and(np.logical_not(sim_mask), gt_mask)
    overlay[tp] = np.array([0.0, 1.0, 0.0])  # verde
    overlay[fp] = np.array([1.0, 0.0, 0.0])  # rojo (sobreestimación)
    overlay[fn] = np.array([0.0, 0.0, 1.0])  # azul (subestimación)
    return overlay


def match_pairs(sim_dir: str, gt_dir: str) -> list[tuple[str, str, str]]:
    pairs = []
    sim_files = sorted([f for f in os.listdir(sim_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    gt_files = set(sim_files)
    for f in sim_files:
        sim_path = os.path.join(sim_dir, f)
        gt_path = os.path.join(gt_dir, f)
        if os.path.exists(gt_path):
            tag = os.path.splitext(f)[0]
            pairs.append((tag, sim_path, gt_path))
    return pairs


def run_validation(sim_dir: str, gt_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pairs = match_pairs(sim_dir, gt_dir)
    rows = ["tag,iou,rel_area_error_percent"]
    ious = []
    errs = []
    for tag, sim_path, gt_path in pairs:
        sim_mask = load_mask(sim_path)
        gt_mask = load_mask(gt_path)
        if sim_mask.shape != gt_mask.shape:
            min_h = min(sim_mask.shape[0], gt_mask.shape[0])
            min_w = min(sim_mask.shape[1], gt_mask.shape[1])
            sim_mask = sim_mask[:min_h, :min_w]
            gt_mask = gt_mask[:min_h, :min_w]
        iou = compute_iou(sim_mask, gt_mask)
        err = compute_relative_area_error(sim_mask, gt_mask)
        ious.append(iou)
        errs.append(err)
        overlay = build_diff_overlay(sim_mask, gt_mask)
        plt.imsave(os.path.join(out_dir, f"{tag}_overlay.png"), overlay)
        rows.append(f"{tag},{iou:.4f},{err:.2f}")
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_err = float(np.mean(errs)) if errs else 0.0
    rows.append(f"MEAN,{mean_iou:.4f},{mean_err:.2f}")
    with open(os.path.join(out_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description="Validación: IoU y Error de Área Relativa entre máscaras")
    parser.add_argument("sim_masks_dir", help="Directorio con máscaras simuladas (PNG binarias)")
    parser.add_argument("gt_masks_dir", help="Directorio con máscaras reales (PNG binarias o segmentadas)")
    parser.add_argument("output_dir", help="Directorio de salida para overlays y CSV")
    args = parser.parse_args()
    run_validation(args.sim_masks_dir, args.gt_masks_dir, args.output_dir)


if __name__ == "__main__":
    main()


