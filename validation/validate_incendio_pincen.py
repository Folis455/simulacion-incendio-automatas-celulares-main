"""
Validacion del Incendio de Pincen - 07/11/2024
===============================================
Coordenadas  : -34.736254674244805, -63.95853104353223
Area real    : 62 ha  (fuente: GIMF/CONAE)
Departamento : General Roca, Buenos Aires, Argentina
Localidad    : PINCEN
Cuenca       : Region Noroeste de la Llanura Pampeana
Cobertura    : Matorral/Arbustal (38%), Pastura natural manejada (35%),
               Zona anegable (21%), Otros (6%)
Pendiente    : 2.1 %  |  Altitud: 145 m.s.n.m  |  Orientacion: Sur
Parcela      : LOTE 12 FC E  |  6 249 833 m2 (~625 ha)

Diferencias con incendio Leguizamon:
  - Noviembre (primavera tardía): temperatura mayor, humedad ambiente mayor
  - Matorral/Arbustal como cobertura dominante: mayor densidad de combustible
  - Zona anegable 21%: se modela como franja de alta humedad de suelo
  - Area 62 ha: grilla 150x150 celdas de 10 m (= 225 ha de cobertura)

Flujo:
  1) Extraer mascara GT desde la imagen de huella catastral (poligono dorado)
  2) Configurar y correr el modelo con parametros calibrados para noviembre 2024
  3) Calcular metricas (IoU, Dice, error de area, Hausdorff)
  4) Generar visualizaciones y CSV de resultados
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mpl_colors

from fire_simulation_model import FireSimulationModel
from config.model_config import GRASS, BURNING, BURNT, EMPTY

# ---------------------------------------------------------------------------
# CONFIGURACION DEL CASO
# ---------------------------------------------------------------------------
GRID_W        = 150          # celdas en X
GRID_H        = 150          # celdas en Y
CELL_SIZE_M   = 10           # metros por celda  -> 150x150 = 225 ha de cobertura
REAL_AREA_HA  = 62.0
TARGET_CELLS  = int(REAL_AREA_HA * 10_000 / (CELL_SIZE_M ** 2))  # 6200 celdas
MAX_STEPS     = 1200

IMAGE_PATH = (
    r"C:\Users\carry\.cursor\projects"
    r"\c-Users-carry-Desktop-2025-simulacion-incendio-automatas-celulares-main"
    r"\assets"
    r"\c__Users_carry_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"547fb677b330b778a7add88aaf7b1d61_images_"
    r"image-1f635361-55bd-4424-8b9c-664431f65614.png"
)

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "OUT_DIR")
SIM_DIR = os.path.join(_HERE, "SIM_DIR")
GT_DIR  = os.path.join(_HERE, "GT_DIR")

# ---------------------------------------------------------------------------
# PASO 1 - Extraer mascara GT desde imagen de huella catastral
# ---------------------------------------------------------------------------

def _neighbor_count(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1, mode="constant")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2]              + p[1:-1, 2:]
        + p[2:, :-2]  + p[2:, 1:-1]  + p[2:, 2:]
    )


def _flood_fill_interior(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Dilata la mascara para cerrar huecos internos del poligono."""
    m = mask.astype(np.uint8)
    for _ in range(iterations):
        p = np.pad(m, 1, mode="constant")
        acc = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2]  + p[2:, 1:-1]  + p[2:, 2:]
        )
        m = np.where(acc >= 2, 1, m).astype(np.uint8)
    return m.astype(bool)


def extract_footprint_mask(image_path: str, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Detecta el poligono dorado/amarillo-oliva de la huella catastral en HSV.
    Devuelve mascara booleana redimensionada a (grid_h, grid_w).
    """
    img = plt.imread(image_path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    rgb = img[..., :3]

    hsv = mpl_colors.rgb_to_hsv(rgb)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    # Amarillo-dorado-oliva: excluye verde (H>0.24), blanco (S<0.22) y azul
    raw_mask = (h >= 0.07) & (h <= 0.24) & (s >= 0.22) & (v >= 0.25)

    m = raw_mask.astype(np.uint8)
    cleaned = (_neighbor_count(m) >= 2) & raw_mask
    filled  = _flood_fill_interior(cleaned, iterations=6)

    in_h, in_w = filled.shape
    yy = np.floor(np.linspace(0, in_h - 1, grid_h)).astype(int)
    xx = np.floor(np.linspace(0, in_w - 1, grid_w)).astype(int)
    return filled[np.ix_(yy, xx)]


# ---------------------------------------------------------------------------
# PASO 2 - Modelo calibrado para incendio Pincen (noviembre 2024)
# ---------------------------------------------------------------------------

def _add_wetland_zones(model: FireSimulationModel, fraction: float = 0.21) -> None:
    """
    Modela la zona anegable (21% del area) como franjas de alta humedad de
    suelo dispersas en la grilla, simulando barreras naturales al fuego.
    La zona anegable NO se apaga automaticamente como agua, pero frena
    la propagacion via mayor humedad de suelo.
    """
    rows, cols = model.grid_size
    rng = np.random.default_rng(seed=42)

    # Generar parches aislados de zona anegable (~21% del area total)
    n_patches = 8
    n_cells_target = int(rows * cols * fraction)
    cells_placed = 0

    for _ in range(n_patches):
        if cells_placed >= n_cells_target:
            break
        cr = rng.integers(int(rows * 0.3), int(rows * 0.85))
        cc = rng.integers(int(cols * 0.1), int(cols * 0.9))
        r_patch = rng.integers(4, 12)
        c_patch = rng.integers(4, 12)

        r0 = max(0, cr - r_patch)
        r1 = min(rows, cr + r_patch)
        c0 = max(0, cc - c_patch)
        c1 = min(cols, cc + c_patch)

        # Alta humedad de suelo en zona anegable (frena pero no bloquea)
        model.dryness_grid[r0:r1, c0:c1] = np.minimum(
            model.dryness_grid[r0:r1, c0:c1], 20.0
        )
        cells_placed += (r1 - r0) * (c1 - c0)


def create_model_pincen(grid_size: tuple = (GRID_H, GRID_W)) -> FireSimulationModel:
    """
    Parametros calibrados para noviembre (primavera tardía) en el NO bonaerense:
      - Temperatura   27 °C  (dia calido de noviembre)
      - Humedad ambt  0.38   (moderada-baja, viento seco del norte)
      - Hum. suelo    0.28   (lluvias de primavera recientes, pero no saturado)
      - Sequedad pasto 65    (primavera: pasto nuevo, aun algo seco)
      - Densidad veget 0.55  (matorral + pastura natural = combustible mas denso)
      - Zona anegable  21%   (parches de dryness baja = menor propagacion)
      - Viento norte -> sur   (coherente con orientacion Sur registrada)
      - Intensidad viento  0.62
    """
    model = FireSimulationModel(grid_size=grid_size)
    rows, cols = grid_size

    # Todo el terreno: mezcla matorral + pastura (GRASS es la abstraccion)
    model.land[:] = GRASS

    # Noviembre en General Roca (Buenos Aires): primavera calida
    model.temperature   = 27.0
    model.humidity      = 0.38
    model.soil_moisture = 0.28
    model.grass_density = 0.55   # matorral + arbustal = combustible denso
    model.dryness_grid  = np.full(grid_size, 65.0, dtype=np.float64)

    # Zona anegable (~21%): parches de baja sequedad que frenan propagacion
    _add_wetland_zones(model, fraction=0.21)

    # Viento del norte: empuja el fuego hacia el sur (orientacion registrada)
    model.wind_direction = [1, 0]
    model.wind_intensity = 0.62

    model.water_grid = np.zeros(grid_size, dtype=np.uint8)
    model.calculate_water_effect()

    # Ignicion en la zona superior de la grilla (norte del poligono)
    # El pin de la imagen aparece en la parte superior-central
    ig_r = int(rows * 0.20)
    ig_c = int(cols * 0.45)
    model.apply_brush(ig_r, ig_c, 4, "fire")

    return model


def run_simulation(
    model: FireSimulationModel,
    target_cells: int,
    max_steps: int = MAX_STEPS,
) -> np.ndarray:
    rows, cols = model.grid_size
    total = rows * cols
    sim_mask = None

    for step in range(max_steps):
        model.update_step()
        burned = int(model.get_burned_mask().sum())

        if step % 100 == 0:
            print(f"    Paso {step:4d}: {burned:6d} celdas quemadas "
                  f"({burned * 100.0 / total:.1f} %)")

        if burned >= target_cells:
            sim_mask = model.get_burned_mask().copy()
            print(f"    -> Area objetivo alcanzada en paso {step} "
                  f"({burned} celdas / {burned * CELL_SIZE_M**2 / 10_000:.1f} ha)")
            break

    if sim_mask is None:
        sim_mask = model.get_burned_mask().copy()
        burned = int(sim_mask.sum())
        print(f"    -> Maximo de pasos alcanzado. Area final: "
              f"{burned} celdas / {burned * CELL_SIZE_M**2 / 10_000:.1f} ha")

    return sim_mask


# ---------------------------------------------------------------------------
# PASO 3 - Metricas
# ---------------------------------------------------------------------------

def compute_iou(sim: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(sim, gt).sum()
    union = np.logical_or(sim, gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def compute_dice(sim: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(sim, gt).sum()
    denom = sim.sum() + gt.sum()
    return (2.0 * inter) / denom if denom > 0 else 1.0


def compute_area_ha(mask: np.ndarray, cell_size_m: float = CELL_SIZE_M) -> float:
    return float(mask.sum()) * cell_size_m ** 2 / 10_000.0


def compute_relative_area_error(sim: np.ndarray, gt: np.ndarray) -> float:
    a_gt = gt.sum()
    return abs(sim.sum() - a_gt) / float(a_gt) * 100.0 if a_gt > 0 else 0.0


def _get_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    interior = (_neighbor_count(m) == 8) & mask
    return np.argwhere(mask & ~interior)


def compute_hausdorff(sim: np.ndarray, gt: np.ndarray) -> float:
    bnd_s = _get_boundary(sim)
    bnd_g = _get_boundary(gt)
    if len(bnd_s) == 0 or len(bnd_g) == 0:
        return float("inf")

    def directed(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        max_min = 0.0
        for pa in pts_a:
            diffs = pts_b - pa
            min_d = float(np.sqrt((diffs ** 2).sum(axis=1)).min())
            if min_d > max_min:
                max_min = min_d
        return max_min

    return max(directed(bnd_s, bnd_g), directed(bnd_g, bnd_s))


def compute_precision_recall(sim: np.ndarray, gt: np.ndarray) -> tuple:
    tp = np.logical_and(sim, gt).sum()
    fp = np.logical_and(sim, ~gt).sum()
    fn = np.logical_and(~sim, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


# ---------------------------------------------------------------------------
# PASO 4 - Visualizacion y reporte
# ---------------------------------------------------------------------------

def plot_results(gt_mask: np.ndarray, sim_mask: np.ndarray, metrics: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "Validacion con Huella - Incendio Pincen  |  07/11/2024\n"
        "Coordenadas: -34.7363, -63.9585  |  Area registrada: 62 ha  |  "
        "Dept. General Roca, Buenos Aires\n"
        "Cobertura: Matorral/Arbustal (38%) + Pastura natural (35%) + Zona anegable (21%)",
        fontsize=9, fontweight="bold",
    )

    axes[0].imshow(gt_mask, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(
        f"Huella Real (GT)\n{metrics['ha_gt']:.1f} ha detectadas en imagen",
        fontsize=9,
    )
    axes[0].axis("off")

    axes[1].imshow(sim_mask, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        f"Simulacion (Automata Celular)\n{metrics['ha_sim']:.1f} ha simuladas",
        fontsize=9,
    )
    axes[1].axis("off")

    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 3), dtype=float)
    tp = gt_mask & sim_mask
    fp = sim_mask & ~gt_mask
    fn = gt_mask & ~sim_mask
    overlay[tp] = [0.15, 0.80, 0.15]
    overlay[fp] = [0.95, 0.15, 0.15]
    overlay[fn] = [0.15, 0.35, 0.95]

    axes[2].imshow(overlay, interpolation="nearest")
    axes[2].set_title(
        "Comparacion celda a celda\nverde=TP  rojo=FP  azul=FN",
        fontsize=9,
    )
    axes[2].axis("off")

    leyenda = [
        mpatches.Patch(color=(0.15, 0.80, 0.15), label="Verdadero Positivo (TP)"),
        mpatches.Patch(color=(0.95, 0.15, 0.15), label="Falso Positivo - sobreestimacion (FP)"),
        mpatches.Patch(color=(0.15, 0.35, 0.95), label="Falso Negativo - subestimacion (FN)"),
    ]
    axes[2].legend(handles=leyenda, loc="lower left", fontsize=7, framealpha=0.88)

    txt_lines = [
        "-- METRICAS ------------------",
        f"Area real (GIMF):    {REAL_AREA_HA:.1f} ha",
        f"Area GT (imagen):    {metrics['ha_gt']:.1f} ha",
        f"Area simulada:       {metrics['ha_sim']:.1f} ha",
        "",
        f"IoU:                 {metrics['iou']:.3f}",
        f"Dice / F1:           {metrics['dice']:.3f}",
        f"Precision:           {metrics['precision']:.3f}",
        f"Recall:              {metrics['recall']:.3f}",
        f"Error de area (GT):  {metrics['area_error']:.1f} %",
        f"Error vs GIMF:       {metrics['area_error_vs_real']:.1f} %",
    ]
    if "hausdorff" in metrics:
        txt_lines.append(f"Hausdorff (celdas):  {metrics['hausdorff']:.1f}")
    txt_lines += [
        "",
        "-- CRITERIOS --",
        f"IoU >= 0.6:    {'OK' if metrics['iou'] >= 0.6 else 'NO'}  ({metrics['iou']:.3f})",
        f"Error <= 15%:  {'OK' if metrics['area_error_vs_real'] <= 15 else 'NO'}  ({metrics['area_error_vs_real']:.1f}%)",
    ]
    if "hausdorff" in metrics:
        txt_lines.append(
            f"Hausdorff<=2:  {'OK' if metrics['hausdorff'] <= 2 else 'NO'}  ({metrics['hausdorff']:.1f})"
        )

    fig.text(
        0.01, 0.01, "\n".join(txt_lines),
        fontsize=7.5, verticalalignment="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.20, 1, 1])
    out_path = os.path.join(OUT_DIR, "validation_pincen.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def write_csv(metrics: dict) -> str:
    csv_path = os.path.join(OUT_DIR, "metrics_pincen.csv")
    header = (
        "incendio,fecha,lat,lon,area_real_ha,area_gt_ha,area_sim_ha,"
        "iou,dice,precision,recall,error_area_vs_gt_pct,error_area_vs_gimf_pct"
    )
    if "hausdorff" in metrics:
        header += ",hausdorff_celdas"
    header += "\n"

    row = (
        f"Pincen,07-11-2024,-34.73625,-63.95853,"
        f"{REAL_AREA_HA:.1f},{metrics['ha_gt']:.2f},{metrics['ha_sim']:.2f},"
        f"{metrics['iou']:.4f},{metrics['dice']:.4f},"
        f"{metrics['precision']:.4f},{metrics['recall']:.4f},"
        f"{metrics['area_error']:.2f},{metrics['area_error_vs_real']:.2f}"
    )
    if "hausdorff" in metrics:
        row += f",{metrics['hausdorff']:.2f}"
    row += "\n"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + row)
    return csv_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SIM_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    sep = "=" * 62

    print(sep)
    print("  VALIDACION INCENDIO PINCEN  -  07-11-2024")
    print(f"  Coordenadas : -34.73625, -63.95853")
    print(f"  Area real   : {REAL_AREA_HA} ha")
    print(f"  Grilla      : {GRID_H}x{GRID_W} celdas @ {CELL_SIZE_M} m/celda")
    print(f"  Cobertura   : Matorral 38% + Pastura 35% + Anegable 21% + Otros 6%")
    print(f"  Objetivo    : {TARGET_CELLS} celdas quemadas")
    print(sep)

    # ---- 1. Mascara GT -------------------------------------------------------
    print("\n[1/4] Extrayendo mascara GT desde imagen de huella catastral ...")
    if not os.path.isfile(IMAGE_PATH):
        print(f"  ADVERTENCIA: imagen no encontrada en:\n  {IMAGE_PATH}")
        print("  Se usara mascara GT sintetica.")
        gt_mask = _synthetic_gt(GRID_H, GRID_W)
    else:
        gt_mask = extract_footprint_mask(IMAGE_PATH, GRID_W, GRID_H)

    gt_cells = int(gt_mask.sum())
    ha_gt    = compute_area_ha(gt_mask)
    print(f"  Celdas GT detectadas : {gt_cells}  ({ha_gt:.1f} ha)")
    plt.imsave(os.path.join(GT_DIR, "t1_pincen_mask.png"),
               gt_mask.astype(np.uint8) * 255, cmap="gray")

    # ---- 2. Configurar modelo ------------------------------------------------
    print("\n[2/4] Configurando modelo (noviembre 2024, matorral + pastura + anegable) ...")
    model = create_model_pincen(grid_size=(GRID_H, GRID_W))
    print(f"  Temperatura     : {model.temperature} C  (primavera calida)")
    print(f"  Humedad ambt    : {model.humidity}")
    print(f"  Hum. suelo      : {model.soil_moisture}")
    print(f"  Densidad veget  : {model.grass_density}  (matorral + pastura)")
    print(f"  Sequedad media  : {model.dryness_grid.mean():.1f}  (zona anegable ~20 en parches)")
    print(f"  Viento          : dir={model.wind_direction}  intensidad={model.wind_intensity}")

    # ---- 3. Simulacion -------------------------------------------------------
    print(f"\n[3/4] Corriendo simulacion (objetivo: {TARGET_CELLS} celdas | max: {MAX_STEPS} pasos) ...")
    sim_mask = run_simulation(model, TARGET_CELLS, MAX_STEPS)
    ha_sim   = compute_area_ha(sim_mask)
    plt.imsave(os.path.join(SIM_DIR, "t1_pincen_mask.png"),
               sim_mask.astype(np.uint8) * 255, cmap="gray")

    # ---- 4. Metricas ---------------------------------------------------------
    print("\n[4/4] Calculando metricas ...")
    iou        = compute_iou(sim_mask, gt_mask)
    dice       = compute_dice(sim_mask, gt_mask)
    area_error = compute_relative_area_error(sim_mask, gt_mask)
    area_error_vs_real = abs(ha_sim - REAL_AREA_HA) / REAL_AREA_HA * 100.0
    precision, recall  = compute_precision_recall(sim_mask, gt_mask)

    metrics = {
        "ha_gt"             : ha_gt,
        "ha_sim"            : ha_sim,
        "iou"               : iou,
        "dice"              : dice,
        "precision"         : precision,
        "recall"            : recall,
        "area_error"        : area_error,
        "area_error_vs_real": area_error_vs_real,
    }

    print("  Calculando distancia de Hausdorff ...")
    try:
        hd = compute_hausdorff(sim_mask, gt_mask)
        metrics["hausdorff"] = hd
    except Exception as e:
        print(f"  Hausdorff omitido: {e}")

    fig_path = plot_results(gt_mask, sim_mask, metrics)
    csv_path = write_csv(metrics)

    # ---- Reporte final -------------------------------------------------------
    print()
    print(sep)
    print("  RESULTADOS")
    print(sep)
    print(f"  Area real (GIMF)       : {REAL_AREA_HA:.1f} ha")
    print(f"  Area GT (imagen)       : {ha_gt:.1f} ha")
    print(f"  Area simulada          : {ha_sim:.1f} ha")
    print()
    print(f"  IoU                    : {iou:.3f}   (criterio >= 0.60)")
    print(f"  Dice / F1              : {dice:.3f}")
    print(f"  Precision              : {precision:.3f}")
    print(f"  Recall                 : {recall:.3f}")
    print(f"  Error de area vs GT    : {area_error:.1f} %")
    print(f"  Error vs area oficial  : {area_error_vs_real:.1f} %  (criterio <= 15 %)")
    if "hausdorff" in metrics:
        print(f"  Hausdorff (celdas)     : {metrics['hausdorff']:.1f}  (criterio <= 2 celdas)")
    print()
    print(f"  Figura  -> {fig_path}")
    print(f"  CSV     -> {csv_path}")
    print(sep)

    passed, failed = [], []
    if iou >= 0.60:
        passed.append(f"IoU {iou:.3f} >= 0.60")
    else:
        failed.append(f"IoU {iou:.3f} < 0.60")

    if area_error_vs_real <= 15.0:
        passed.append(f"Error area vs GIMF {area_error_vs_real:.1f}% <= 15%")
    else:
        failed.append(f"Error area vs GIMF {area_error_vs_real:.1f}% > 15%")

    if "hausdorff" in metrics:
        if metrics["hausdorff"] <= 2.0:
            passed.append(f"Hausdorff {metrics['hausdorff']:.1f} <= 2 celdas")
        else:
            failed.append(f"Hausdorff {metrics['hausdorff']:.1f} > 2 celdas")

    if failed:
        print(f"\n  [NO] Criterios NO alcanzados: {'; '.join(failed)}")
    if passed:
        print(f"  [OK] Criterios alcanzados   : {'; '.join(passed)}")
    print()


def _synthetic_gt(rows: int, cols: int) -> np.ndarray:
    """Mascara sintetica de respaldo: forma irregular orientada al sur."""
    mask = np.zeros((rows, cols), dtype=bool)
    cr, cc = int(rows * 0.45), int(cols * 0.45)
    ra, rb = int(rows * 0.35), int(cols * 0.28)
    for r in range(rows):
        for c in range(cols):
            if ((r - cr) / ra) ** 2 + ((c - cc) / rb) ** 2 <= 1.0:
                mask[r, c] = True
    return mask


if __name__ == "__main__":
    main()
