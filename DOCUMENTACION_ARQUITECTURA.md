# Documentación de Arquitectura: Simulación de Incendios con Autómatas Celulares

## Índice

1. [Visión General del Sistema](#visión-general-del-sistema)
2. [Arquitectura del Modelo (fire_simulation_model.py)](#arquitectura-del-modelo)
3. [Arquitectura de la GUI (fire_simulation_gui.py)](#arquitectura-de-la-gui)
4. [Decisiones de Diseño](#decisiones-de-diseño)
5. [Variables y Parámetros Críticos](#variables-y-parámetros-críticos)
6. [Flujo de Ejecución](#flujo-de-ejecución)

---

## Visión General del Sistema

El sistema implementa una simulación de incendios forestales basada en **autómatas celulares estocásticos** usando cadenas de
Markov. La arquitectura está dividida en dos componentes principales:

- **Modelo de Simulación** (`fire_simulation_model.py`): Lógica de negocio y matemática
- **Interfaz Gráfica** (`fire_simulation_gui.py`): Visualización e interacción del usuario

### Principios Arquitectónicos Aplicados

- **Separación de Responsabilidades**: Modelo y vista completamente separados
- **Programación Orientada a Objetos**: Encapsulación de funcionalidad en clases
- **Patrón MVC**: Modelo-Vista-Controlador adaptado para simulaciones científicas

---

## Arquitectura del Modelo (fire_simulation_model.py)

### Estados del Sistema

```python
EMPTY = 0  # Celda vacía
GRASS = 1  # Pasto (anteriormente TREE)
BURNING = 2  # En combustión
BURNT = 3  # Quemado
```

**Decisión de Diseño**: Se usaron constantes numéricas para optimizar el rendimiento en arrays NumPy, evitando comparaciones de
strings.

### Clase FireSimulationModel

#### Propósito y Responsabilidades

- **Gestión del Estado**: Mantiene la cuadrícula principal y grids auxiliares
- **Lógica de Transición**: Implementa las reglas de transición entre estados
- **Simulación Física**: Modela efectos climáticos y propagación del fuego

#### Atributos Principales

```python
self.land: np.array  # Cuadrícula principal de estados
self.dryness_grid: np.array  # Nivel de sequedad por celda [0-100]
self.water_grid: np.array  # Presencia de agua por celda
```

**Decisión**: Se separaron en grids independientes para:

- Flexibilidad en el modelado
- Mantener actualizaciones sincronizadas: ninguna celda ve cambios parciales de sus vecinas dentro del mismo paso.
- Optimización de memoria (diferentes tipos de datos)
- Facilidad de debugging y visualización
- Correctitud/determinismo del autómata: todas las reglas se aplican sobre el mismo estado base.
- \+ Correctitud y resultados reproducibles.
- − Más memoria y tiempo de copia por paso.

**Alternativas:**

- Mantener dos arrays fijos y alternarlas (swap) en lugar de copy() -> Posibles bugs de concurrencia, el costo del copy es
  marginal para los tamaños de grid actuales.

- Actualizar in-place solo si lees siempre de un buffer inmutable (más complejo y propenso a errores).

#### Variables Climáticas

```python
self.temperature: float  # Temperatura ambiente
self.soil_moisture: float  # Humedad del suelo [0-1]
self.wind_direction: list  # [componente_y, componente_x]
self.wind_intensity: float  # Intensidad del viento [0-1]
self.humidity: float  # Humedad del aire [0-1]
```

**Decisión**: Variables continuas para modelado realista, permitiendo interpolación suave de efectos.

### Función: get_neighborhood()

#### Propósito

Obtiene los 8 vecinos de una celda según la **vecindad de Moore**.

#### Decisiones de Implementación

```python
# Manejo de bordes como EMPTY
if nr < 0 or nc < 0 or nr >= grid.shape[0] or nc >= grid.shape[1]:
    neighbors.append(EMPTY)
```

**Justificación**: Los bordes como vacío simulan "condiciones de frontera abiertas", más realista que bordes periódicos.

#### Orden de Vecinos

```python
NEIGHBOR_RELATIVE_COORDS = [
    (-1, -1), (-1, 0), (-1, 1),  # Fila superior
    (0, -1), (0, 1),  # Fila central (sin centro)
    (1, -1), (1, 0), (1, 1)  # Fila inferior
]
```

**Decisión**: Orden fijo y documentado para cálculos consistentes de viento direccional.

### Función: get_transition_matrix()

#### Arquitectura de Probabilidades

La función implementa un **sistema híbrido**:

1. **Matriz base**: Probabilidades por defecto
2. **Modificadores dinámicos**: Efectos climáticos y ambientales

#### Matriz de Transición Base

```python
base_transitions = np.array([
    [1.0 - P_E_G_BASE, P_E_G_BASE, 0.00, 0.00],  # EMPTY
    [0.00, 1.0 - P_G_B_BASE, P_G_B_BASE, 0.00],  # GRASS
    [0.00, 0.00, P_B_B, P_B_X],  # BURNING
    [P_X_E_BASE, 0.00, 0.00, 1.0 - P_X_E_BASE]  # BURNT
])
```

**Decisión**: Matriz estructurada para garantizar:

- Transiciones físicamente posibles
- Conservación de probabilidad (suma = 1)
- Fácil modificación de parámetros

#### Cálculo de Efectos Climáticos

##### Efecto del Viento

```python
# Producto escalar para determinar alineación
dot_product = (-dr_n * wind_y_comp) + (-dc_n * wind_x_comp)
alignment = dot_product / (norm_neighbor_vec * norm_wind)
```

**Justificación Matemática**:

- Usa álgebra vectorial para calcular cuánto "favorece" el viento la propagación
- Valores positivos = viento a favor, negativos = en contra
- Normalización para independence de la magnitud

##### Efectos de Humedad

```python
humidity_reduction_factor = 1.0 - hum
soil_moisture_reduction_factor = 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY)
```

**Decisión**: Factores multiplicativos lineales por simplicidad y interpretabilidad.

##### Efecto de Temperatura

```python
if temp_c > TEMP_BASELINE_C:
    temp_effect_on_spread = (temp_c - TEMP_BASELINE_C) * TEMP_SENSITIVITY
```

**Decisión**: Solo temperaturas sobre baseline afectan (realismo físico).

##### Efecto de Sequedad Local

```python
dryness_scale = 1.0 + (dryness_norm) * DRYNESS_SPREAD_MULTIPLIER
```

**Decisión**: Factor multiplicativo escalado para amplificar probabilidad de propagación del fuego en áreas secas.

### Función: update_step()

#### Arquitectura de Actualización

Implementa el **paradigma de doble buffer**:

```python
new_grid = self.land.copy()  # Buffer secundario
# ... cálculos en new_grid ...
self.land = new_grid  # Intercambio atómico
```

**Justificación**: Evita dependencias temporales y garantiza determinismo.

#### Tratamiento Especial del Agua

```python
if self.water_grid[r, c] > 0:
    current_cell_state = EMPTY
    # ... luego forzar resultado a EMPTY
```

**Decisión**: Agua como "override absoluto" - simplifica lógica y es físicamente realista.

### Parámetros de Calibración

#### Probabilidades Base (Muy Reducidas)

```python
P_E_G_BASE = 0.00000  # Sin regeneración automática
P_G_B_BASE = 0.00000  # Sin ignición espontánea
P_X_E_BASE = 0.00000  # Sin regeneración de quemado
```

**Decisión Crítica**: Valores en cero fuerzan que toda dinámica provenga de:

- Interacciones entre vecinos
- Efectos climáticos
- Intervención del usuario

#### Factores de Propagación

```python
F_G_B_NEIGHBOR = 0.15  # Propagación por vecinos
F_G_B_WIND = 0.35  # Amplificación por viento
P_G_B_MAX = 0.95  # Límite máximo
```

**Calibración**: Valores ajustados empíricamente para:

- Propagación realista (no demasiado rápida/lenta)
- Efecto visible del viento
- Prevenir probabilidades > 1.0 (Esto es para permitir que no haya probabilidades de incendio al 100% y mantener aleatoriedad)

#### Duración de Combustión

```python
P_B_B = 0.80  # 80% probabilidad de seguir quemando
P_B_X = 0.20  # 20% probabilidad de extinguirse
```

**Decisión**: Ratio 4:1 para duración promedio de 5 pasos, balanceando realismo y velocidad de simulación.

---

## Arquitectura de la GUI (fire_simulation_gui.py)

### Clase FireSimulationGUI

#### Principios de Diseño

- **Patrón Observer**: GUI observa cambios en el modelo
- **Separación de Responsabilidades**: GUI solo maneja visualización e interacción
- **Programación por Eventos**: Respuesta a acciones del usuario

#### Estructura de Datos

##### Variables de Estado de Simulación

```python
self.simulation_started: bool
self.simulation_paused: bool
self.ani: matplotlib.animation
self.stats_history: list
```

**Decisión**: Estado explícito para manejar ciclo de vida de la simulación.

##### Variables de Interacción

```python
self.brush_size_cells: int
self.brush_dryness_value: float
self.painting_mode: str
self.show_water_overlay: bool
```

**Diseño**: Configuración persistente del "pincel" para edición fluida del escenario.

### Función: _setup_ui()

#### Layout de la Interfaz

```python
self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(17, 8))
```

**Decisión**: División 1:2 para balancear visualización de simulación y estadísticas.

#### Panel de Estadísticas

```python
self.line_empty, = self.ax2.plot([], [], color='lightgray', ...)
self.line_grass, = self.ax2.plot([], [], color=colors[GRASS], ...)
self.line_burning, = self.ax2.plot([], [], color=colors[BURNING], ...)
self.line_burnt, = self.ax2.plot([], [], color=colors[BURNT], ...)
```

**Decisión**: Una línea por estado para tracking temporal independiente.

### Función: _build_display_image()

#### Sistema de Colores

##### Colores Base por Estado

```python
color_empty = np.array([1.0, 1.0, 1.0])  # Blanco
color_green = np.array([0.0, 0.5, 0.0])  # Verde oscuro
color_yellow = np.array([1.0, 1.0, 0.0])  # Amarillo
color_red = np.array([1.0, 0.0, 0.0])  # Rojo
color_blue = np.array([0.0, 0.4, 1.0])  # Azul
color_black = np.array([0.0, 0.0, 0.0])  # Negro
```

**Justificación**: Colores intuitivos y contrastantes para fácil interpretación.

#### Degradado de Sequedad para Pasto

```python
# Interpolación verde -> amarillo basada en sequedad
w_yellow = np.power(dryness_norm[grass_idx], 1.5)
w_green = 1.0 - w_yellow
img[grass_idx] = (w_green[:, None] * color_green) + (w_yellow[:, None] * color_yellow)
```

**Decisiones**:

- **Exponente 1.5**: Curva no-lineal para resaltar diferencias visuales
- **Interpolación lineal**: Entre verde (húmedo) y amarillo (seco)
- **Broadcasting NumPy**: Para eficiencia computacional

#### Overlay de Agua

```python
if self.show_water_overlay:
    # Mostrar área de influencia del agua
    pad = WATER_EFFECT_RADIUS
    # ... cálculo de área de influencia ...
    alpha = 0.25
    img[overlay_mask] = (1 - alpha) * img[overlay_mask] + alpha * cyan
```

**Diseño**:

- **Opcional**: No interfiere con visualización normal
- **Alpha blending**: Superpone información sin ocultar completamente
- **Radio de efecto**: Visualiza mecánica interna del modelo

### Sistema de Controles

#### Sliders para Parámetros Climáticos

```python
self.slider_wind_x = Slider(ax_wind_x, 'Viento X', -10.0, 10.0, ...)
self.slider_wind_y = Slider(ax_wind_y, 'Viento Y', -10.0, 10.0, ...)
```

**Decisión**: Rangos simétricos [-10, 10] para permitir vientos en cualquier dirección.

#### Callbacks de Actualización

```python
def _update_wind_x(self, val):
    self.model.wind_direction[1] = val
```

**Arquitectura**: Callbacks directos al modelo para actualizaciones en tiempo real.

### Sistema de Eventos

#### Manejo de Mouse

```python
def _on_click(self, event):
    if event.inaxes != self.ax1: return  # Solo en panel de simulación


def _on_hover(self, event):
# Resaltado + pintado mientras se arrastra
```

**Decisiones**:

- **Verificación de área**: Solo activo en panel de simulación
- **Estados de mouse**: Distingue entre hover, click y drag
- **Feedback visual**: Resaltado inmediato de celdas

#### Hotkeys de Teclado

```python
if event.key == 'm':  # Cambiar modo de pincel
    if event.key == '.':  # Aumentar tamaño de pincel
        if event.key == ',':  # Disminuir tamaño de pincel
            if event.key in '1234567890':  # Sequedad rápida
```

**Justificación**: Atajos para edición eficiente sin apartar manos del mouse.

### Función: _update_animation()

#### Arquitectura de Animación

**Decisiones**:

- **Velocidad variable**: Ejecuta múltiples pasos por frame, pero **AUMENTA el delay** entre frames
- **Estadísticas por paso**: Mantiene granularidad de datos pero reduce fluidez visual
- **Historia persistente**: Para análisis temporal

**Problema de Rendimiento Identificado**:

- La multiplicación de pasos causa delays proporcionales (5x = 5 veces más lento)
- No es verdadero "fast-forward", sino "skip frames"
- La animación se vuelve menos fluida a velocidades altas

#### Actualización de Gráficos

```python
self.line_empty.set_data(x_data, [s['empty'] for s in self.stats_history])
```

**Arquitectura**: Reconstrucción completa de datos en cada frame (simple pero ineficiente para simulaciones largas).

---

## Decisiones de Diseño

### 1. Arquitectura Modular

**Decisión**: Separar completamente modelo y vista
**Justificación**:

- Permite testing independiente
- Facilita cambios de interfaz
- Reutilización del modelo en diferentes contextos

### 2. Uso de NumPy

**Decisión**: Arrays NumPy para toda la computación numérica
**Justificación**:

- Rendimiento optimizado para operaciones vectoriales
- Interfaz limpia con matplotlib
- Memoria eficiente para grids grandes

### 3. Parámetros Configurables

**Decisión**: Constantes globales para todos los parámetros físicos
**Justificación**:

- Fácil calibración sin recompilación
- Documentación centralizada
- Experimentación rápida

### 4. Sistema de Estados Determinista

**Decisión**: Estados discretos con transiciones probabilísticas
**Justificación**:

- Simplifica visualización
- Facilita análisis estadístico
- Comportamiento predecible

### 5. Doble Buffer para Actualización

**Decisión**: Copiar grid completo en cada paso
**Justificación**:

- Garantiza determinismo
- Evita condiciones de carrera
- Simplifica debugging

---

## Variables y Parámetros Críticos

### Parámetros de Calibración Física

| Variable                    | Valor | Propósito                  | Justificación                    |
|-----------------------------|-------|----------------------------|----------------------------------|
| `F_G_B_NEIGHBOR`            | 0.15  | Propagación entre vecinos  | Balance realismo/velocidad       |
| `F_G_B_WIND`                | 0.35  | Amplificación por viento   | Efecto visible pero no dominante |
| `P_B_B`                     | 0.80  | Duración de combustión     | ~5 pasos promedio                |
| `TEMP_BASELINE_C`           | 20.0  | Temperatura neutra         | Temperatura ambiente típica      |
| `SOIL_MOISTURE_SENSITIVITY` | 0.7   | Efecto humedad suelo       | Efecto fuerte pero no absoluto   |
| `DRYNESS_SPREAD_MULTIPLIER` | 0.8   | Amplificación por sequedad | Efecto notable de sequedad       |
| `WATER_EFFECT_RADIUS`       | 3     | Radio influencia agua      | Balance localidad/efecto         |

### Variables de Estado Críticas

| Variable         | Tipo              | Rango   | Propósito                      |
|------------------|-------------------|---------|--------------------------------|
| `forest`         | `np.array[int]`   | [0-3]   | Estado principal de simulación |
| `dryness_grid`   | `np.array[float]` | [0-100] | Sequedad local por celda       |
| `water_grid`     | `np.array[uint8]` | [0-1]   | Presencia de agua              |
| `wind_direction` | `list[float]`     | [-∞,∞]  | Vector viento [y,x]            |
| `wind_intensity` | `float`           | [0-1]   | Magnitud del viento            |

---

## Flujo de Ejecución

### 1. Inicialización

```
FireSimulationGUI.__init__()
├── Crear FireSimulationModel
├── _setup_ui() → Configurar matplotlib
├── _setup_controls() → Crear sliders/botones
├── _setup_event_handlers() → Conectar eventos
└── update_brush_text() → Estado inicial UI
```

### 2. Bucle de Simulación

```
_main_button_callback() → Iniciar
├── Verificar/Crear fuego inicial
├── animation.FuncAnimation(_update_animation)
└── Para cada frame:
    ├── model.update_step() × velocidad
    ├── Actualizar estadísticas
    ├── _build_display_image()
    └── Actualizar gráficos
```

### 3. Paso de Simulación Individual

```
model.update_step()
├── Para cada celda (r,c):
│   ├── get_neighborhood() → Estados vecinos
│   ├── get_transition_matrix() → Probabilidades
│   │   ├── Matriz base
│   │   ├── Efectos climáticos
│   │   ├── Efecto viento direccional
│   │   └── Normalización
│   └── np.random.choice() → Nuevo estado
└── Intercambio de buffers
```

### 4. Interacción del Usuario

```
Eventos de Mouse/Teclado
├── _on_click() → Aplicar pincel
├── _on_hover() → Resaltado + drag
├── _on_key() → Hotkeys
└── Slider callbacks → Actualizar parámetros
```

---

## Conclusiones Arquitectónicas

### Fortalezas del Diseño

1. **Modularidad**: Separación clara de responsabilidades
2. **Extensibilidad**: Fácil agregar nuevos parámetros o estados
3. **Performance**: Uso eficiente de NumPy para cálculos vectoriales
4. **Usabilidad**: Interfaz intuitiva con feedback visual inmediato
5. **Configurabilidad**: Parámetros ajustables sin recompilación

### Áreas de Mejora Potencial

1. **Análisis**: Agregar IA para realizar análisis y ajuste de variables.
2. **Validación**: Agregar IA para contrastar y validar datos.
3. **Imágenes satelitales**: Realizar importación de imágenes satelitales a grids una vez se hagan ajustes de validación y
   variables.
4. **Velocidad**: Optimizar velocidad de cómputo para incrementar la velocidad de la animación de la simulación.

### Decisiones

- **Simplicidad vs Performance**: Se priorizó código legible y sencillo sobre optimización extrema. Se decidió dejar
  implementaciones de IA como plausibles mejoras futuras.
- **Importación satelital vs Diseño de usuario**: El terreno se diseña por el usuario en vez de trabajar con imágenes satélitales
  para garantizar un modelo funcional para testeos que luego podrá ser contrastado con resultados satelitales.

Esta arquitectura proporciona una base sólida para simulaciones de incendios educativas y de investigación, balanceando realismo
científico con usabilidad práctica.
