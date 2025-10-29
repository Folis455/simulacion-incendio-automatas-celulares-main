# Validación de la simulación contra datos reales (paso a paso reproducible)

Este README describe un flujo breve y reproducible usando capturas de NASA FIRMS y el preset HSV ya calibrado para generar máscaras reales y compararlas con la simulación.

## Requisitos
- Python 3.10+
- `pip install -r requirements.txt`
- Utilidades: `validation.py`, `firms_to_mask.py` (incluidos en el repo)

## Estructura de carpetas
```
validation/
  SIM_DIR/   # máscaras simuladas (t1_mask.png, t2_mask.png, ...)
  GT_DIR/    # máscaras reales procesadas (t1_mask.png, t2_mask.png, ...)
  OUT_DIR/   # métricas y overlays
```

## 1) Exportar snapshots desde la GUI
1. `python run.py`
2. Configurar t0 (agua, pasto, sequedad, clima) según imagen real.
3. Simular hasta t1, Pausa, botón "Guardar Snapshot" → genera `t1_view.png`, `t1_mask.png`, `t1.npz`.
4. Repetir para t2, t3...
5. Copiar a `validation/SIM_DIR/` y renombrar a `t*_mask.png` si hace falta.

Ejemplo (PowerShell):
```powershell
New-Item -ItemType Directory -Force -Path ".\validation\SIM_DIR" | Out-Null
New-Item -ItemType Directory -Force -Path ".\validation\GT_DIR"  | Out-Null
New-Item -ItemType Directory -Force -Path ".\validation\OUT_DIR" | Out-Null
Copy-Item "C:\Ruta\snapshot_t1_mask.png" ".\validation\SIM_DIR\t1_mask.png" -Force
```

## 2) Construir máscaras reales desde NASA FIRMS (modo CAPTURE)
1. Abrir `https://firms.modaps.eosdis.nasa.gov/map/`.
2. Ir a la zona del incendio y seleccionar el intervalo de fechas.
3. Usar el botón "CAPTURE" para descargar una imagen JPG del mapa con los hotspots visibles.
4. Convertir esa captura a máscara 0/1 con el preset HSV ya ajustado:
```powershell
python .\firms_to_mask.py "C:\ruta\FIRMS.jpg" ".\validation\GT_DIR\t1_mask.png" `
  --width 100 --height 100 `
  --preview ".\validation\GT_DIR\t1_preview.png" `
  --mode hsv --min_saturation 0.70 --min_neighbors 4
```
La imagen `t1_preview.png` permite verificar visualmente la detección; `t1_mask.png` es la máscara binaria compatible con `validation.py`.

## 3) Ejecutar la validación
```powershell
python .\validation.py ".\validation\SIM_DIR" ".\validation\GT_DIR" ".\validation\OUT_DIR"
```
Salida:
- `OUT_DIR/metrics.csv` (IoU y Error de área por tiempo y promedios)
- `OUT_DIR/*_overlay.png` (verde=acierto, rojo=sobreestimación, azul=subestimación)

## 4) Parametrizaciones por defecto
- Cuadrícula: usar `DEFAULT_GRID_SIZE` (por defecto 100×100) en `--width/--height`.
- FIRMS preset fijo (HSV): `--mode hsv --min_saturation 0.70 --min_neighbors 4`.

## 5) Escalar a otros incendios
- Repetir exactamente el mismo flujo con nuevas capturas FIRMS (CAPTURE) y las mismas opciones HSV y tamaño de grilla.
- Mantener nombres `t*_mask.png` en ambas carpetas para que `validation.py` empareje correctamente.


