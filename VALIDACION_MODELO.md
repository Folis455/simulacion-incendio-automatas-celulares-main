# Plan de Validación del Modelo de Incendios

## Objetivo
Validar que la simulación reproduzca la evolución espacial y temporal de un incendio real a partir de imágenes en distintos momentos. Se construirá un mapa equivalente (pintado sobre la retícula), se simulará bajo condiciones similares y se contrastará en tiempos alineados.

## Insumos necesarios
- Imágenes del mismo incendio en tiempos t0, t1, t2, … (idealmente con escala o georreferenciadas)
- Datos ambientales aproximados por intervalo: viento (dirección/intensidad), temperatura, humedad de aire/suelo, precipitaciones
- Segmentación o delimitación de áreas: agua, combustible (pasto), zonas quemadas y, si es posible, frente activo

## Preparación del mapa (pintado en la grilla)
1) Definir tamaño de grilla y resolución (ej.: 100×100 celdas) acorde a la escala de la imagen
2) Calibrar la escala: 1 celda ≈ X metros (mantener fija en todos los tiempos)
3) Pintar sobre la grilla:
   - Agua en `water_grid`
   - Combustible (pasto) en `forest == GRASS`
   - Sequedad local en `dryness_grid` (si hay variación espacial)
   - Estado inicial (quemándose/quemado) según la imagen t0, si corresponde
4) Registrar parámetros climáticos iniciales (viento, temperatura, humedades)

## Protocolo paso a paso
1) Reunir imágenes en t0 < t1 < t2 < …
2) Construir el mapa inicial en t0 pintando la grilla para que coincida con terreno y estado del incendio
3) Configurar parámetros climáticos promedio entre t0→t1
4) Ejecutar la simulación desde t0 hasta t1 (mismo tiempo simulado que el real)
5) Guardar el estado simulado en t1 (snapshot S1) y comparar con la imagen real I1
6) Repetir para cada intervalo ti→ti+1 ajustando parámetros si hay evidencia de cambios (viento, lluvia, etc.)
7) Registrar resultados y métricas para cada tiempo comparado

## Alineación temporal
- Usar la diferencia real (horas/minutos) para mapear a pasos de simulación
- Mantener una razón fija “pasos de simulación por hora real” o ajustarla por intervalo si hay variaciones fuertes

## Qué comparar en cada tiempo
- Máscara de área quemada (BURNT) y, si es posible, zona en combustión (BURNING)
- Contorno del frente de incendio (si puede extraerse del dato real)

## Métricas recomendadas
Elegir 1–3 como principales y usar las restantes como apoyo.

1) Superposición de áreas
- IoU (Intersection over Union) entre máscara quemada simulada y real
- Dice/F1 para máscaras: 2·|A∩B| / (|A|+|B|)
- Error de área relativa: |A_sim − A_real| / A_real

2) Distancia de frentes
- Distancia Hausdorff entre contornos (máxima discrepancia de borde)
- Distancia media de contornos (Average Symmetric Surface Distance) para robustez

3) Evolución temporal
- RMSE de “área quemada acumulada” a lo largo del tiempo (curva simulada vs real)
- Desfase temporal (lag) para alcanzar hitos de porcentaje quemado (p.ej., 20%, 50%, 80%)

4) Celda a celda (si la georreferenciación es buena)
- Matriz de confusión por celda (TP/FP/FN/TN) sobre quemado/no quemado
- Precisión y recall del área quemada

Notas:
- IoU/Dice evalúan solapamiento; Hausdorff evalúa error de borde; RMSE temporal evalúa la dinámica
- Si las imágenes no distinguen “quemándose” de “quemado”, comparar solo la máscara de quemado acumulado

## Pipeline de cálculo de métricas
1) Preprocesar imagen real: escalar a la grilla y segmentar máscaras (agua, combustible, quemado)
2) Rasterizar a la resolución de la simulación (mismo ancho×alto)
3) Obtener del simulador la(s) máscara(s) correspondiente(s) para el mismo tiempo
4) Calcular las métricas seleccionadas
5) Repetir por cada tiempo y promediar/reportar estadísticas agregadas (media, desvío, percentiles)

## Reporte y criterios de aceptación (sugeridos)
- Por tiempo: IoU, distancia de frentes, área quemada simulada vs real
- Resumen global: media y desvío de IoU; RMSE de la curva de área; percentiles de distancia de frentes
- Criterios ejemplo (ajustables): IoU medio ≥ 0.6; distancia media de frentes ≤ 2 celdas; RMSE de área ≤ 15% del área real máxima

## Registro para reproducibilidad
- Por intervalo: viento (dir/intensidad), temperatura, humedad aire/suelo, lluvia
- Ajustes manuales (pincel, agua añadida) y multiplicadores de sequedad
