import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors


def load_image_rgba(path: str) -> np.ndarray:
    img = plt.imread(path)
    # img puede venir como uint8 0..255 o float 0..1 y con 3 o 4 canales
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.shape[-1] == 3:
        # agregar alpha=1
        h, w, _ = img.shape
        alpha = np.ones((h, w, 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=-1)
    return img.astype(np.float32)


def rgba_to_hsv(rgb: np.ndarray) -> np.ndarray:
    # rgb expected in 0..1, shape (..., 3)
    return mpl_colors.rgb_to_hsv(rgb)


def threshold_firms_hotspots(
    rgba: np.ndarray,
    mode: str = "redscore",
    top_percent: float = 2.0,
    min_redness: float = 0.25,
    min_saturation: float = 0.5,
    min_neighbors: int = 3,
    use_alpha: bool = False,
) -> np.ndarray:
    """
    Extrae hotspots FIRMS aproximando por color (rojo/naranja) y/o alpha elevado.
    Devuelve máscara booleana (True = hotspot/quemado).
    """
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    hsv = rgba_to_hsv(rgb)
    s = hsv[..., 1]

    if mode == "hsv":
        h = hsv[..., 0]
        v = hsv[..., 2]
        is_red1 = (h <= 0.06)  # ~0-22°
        is_red2 = (h >= 0.92)  # ~330-360°
        is_orange = (h >= 0.06) & (h <= 0.15)  # ~22-54°
        high_sat = s >= max(0.35, min_saturation)
        bright = v >= 0.25
        color_mask = (is_red1 | is_red2 | is_orange) & high_sat & bright
    else:
        # "redscore": seleccionar solo el percentil superior de rojez
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        redness = r - np.maximum(g, b)  # en [−1,1], pero útil ~[0,1]
        # evitar tonos desaturados
        redness[s < min_saturation] = -1.0
        thr = np.quantile(redness, max(0.0, 1.0 - top_percent / 100.0))
        thr = max(thr, min_redness)
        color_mask = redness >= thr

    # Si la captura preserva alpha del overlay y se habilita, usarlo para reforzar
    if use_alpha:
        alpha_mask = alpha >= 0.25
        mask = color_mask | alpha_mask
    else:
        mask = color_mask

    # Pequeña limpieza: quita ruido de píxeles aislados con un conteo 3x3
    m = mask.astype(np.uint8)
    # Convolución manual simple (sin scipy):
    padded = np.pad(m, 1, mode='constant', constant_values=0)
    acc = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:  , 0:-2] + padded[2:  , 1:-1] + padded[2:  , 2:]
    )
    # Mantener píxeles con >= min_neighbors vecinos en 3x3 (reduce manchas sueltas)
    cleaned = (acc >= int(min_neighbors))
    return cleaned


def resize_mask(mask: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None or height is None:
        return mask
    # nearest neighbor manual
    in_h, in_w = mask.shape
    yy = (np.floor(np.linspace(0, in_h - 1, height)).astype(int))
    xx = (np.floor(np.linspace(0, in_w - 1, width)).astype(int))
    return mask[np.ix_(yy, xx)]


def save_mask_png(mask: np.ndarray, out_path: str) -> None:
    plt.imsave(out_path, mask.astype(np.uint8), cmap='gray', vmin=0, vmax=1)


def save_preview_overlay(rgba: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    base = rgba[..., :3]
    overlay = np.copy(base)
    # Pintar máscara en rojo semitransparente
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.35
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * red
    plt.imsave(out_path, np.clip(overlay, 0, 1))


def main():
    p = argparse.ArgumentParser(description="Convierte captura FIRMS (hotspots) en máscara binaria para validation.py")
    p.add_argument("input_image", help="Ruta a la imagen FIRMS (JPG/PNG)")
    p.add_argument("output_mask", help="Ruta de salida para la máscara (PNG binario)")
    p.add_argument("--width", type=int, default=None, help="Ancho destino (p. ej. 100, igual a grilla)")
    p.add_argument("--height", type=int, default=None, help="Alto destino (p. ej. 100, igual a grilla)")
    p.add_argument("--preview", default=None, help="PNG de vista previa con la máscara sobre la imagen")
    p.add_argument("--mode", choices=["redscore", "hsv"], default="redscore", help="Estrategia de detección (redscore recomendado)")
    p.add_argument("--top_percent", type=float, default=2.0, help="Top % más rojos a conservar (redscore)")
    p.add_argument("--min_redness", type=float, default=0.25, help="Rojez mínima absoluta 0-1 (redscore)")
    p.add_argument("--min_saturation", type=float, default=0.5, help="Saturación mínima 0-1")
    p.add_argument("--min_neighbors", type=int, default=3, help="Vecinos 3x3 mínimos para conservar un píxel")
    p.add_argument("--use_alpha", action="store_true", help="Usar canal alpha del overlay si existe")
    args = p.parse_args()

    rgba = load_image_rgba(args.input_image)
    mask = threshold_firms_hotspots(
        rgba,
        mode=args.mode,
        top_percent=args.top_percent,
        min_redness=args.min_redness,
        min_saturation=args.min_saturation,
        min_neighbors=args.min_neighbors,
        use_alpha=args.use_alpha,
    )
    mask = resize_mask(mask, args.width, args.height)
    save_mask_png(mask, args.output_mask)
    # Reporte de cobertura para ajuste rápido
    h, w = mask.shape
    coverage = float(mask.mean()) * 100.0
    print(f"[firms_to_mask] size={w}x{h} coverage={coverage:.3f}%")
    if args.preview:
        # Si se redimensionó, también crear overlay del tamaño final
        if args.width is not None and args.height is not None:
            # redimensionar base con bilinear simple (usamos matplotlib image resample via imshow-grid? implement NN para simplicidad)
            in_h, in_w = rgba.shape[:2]
            yy = (np.floor(np.linspace(0, in_h - 1, args.height)).astype(int))
            xx = (np.floor(np.linspace(0, in_w - 1, args.width)).astype(int))
            base_resized = rgba[np.ix_(yy, xx, np.arange(4))]
            save_preview_overlay(base_resized, mask, args.preview)
        else:
            save_preview_overlay(rgba, mask, args.preview)


if __name__ == "__main__":
    main()


# python firms_to_mask.py "C:\ruta\a\firms.jpg" "C:\ruta\GT_DIR\t1_mask.png" --width 100 --height 100 --preview "C:\ruta\GT_DIR\t1_preview.png"