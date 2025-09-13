import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
from matplotlib.colors import ListedColormap # type: ignore
from matplotlib.widgets import Slider, Button # type: ignore
import matplotlib.patches as patches # type: ignore # Importado para el resaltado
import math

# Estados de la celda
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3

# Parámetros de simulación
GRID_SIZE = (100, 100)  # Tamaño de la cuadrícula
SIMULATION_STEPS = 50 # Número de pasos de la simulación

# Probabilidades base y factores de influencia (para get_transition_matrix)
# EMPTY state transitions
P_E_T_BASE = 0.00000  # Probabilidad base EMPTY -> TREE (REDUCIDO DRÁSTICAMENTE de 0.02)
F_E_T_NEIGHBOR = 0.00  # Factor de influencia de vecinos TREE para EMPTY -> TREE 
P_E_T_MAX_INFLUENCE = 0.00  # Máxima probabilidad adicional por influencia para EMPTY -> TREE (REDUCIDO de 0.3)

# TREE state transitions
P_T_B_BASE = 0.00000  # Probabilidad base TREE -> BURNING (REDUCIDO DRÁSTICAMENTE de 0.05)
F_T_B_NEIGHBOR = 0.15  # Factor de propagación por vecinos BURNING para TREE -> BURNING
F_T_B_WIND = 0.35  # Factor de escala para la influencia del viento en TREE -> BURNING (AUMENTADO de 0.1)
P_T_B_MAX = 0.95  # Máxima probabilidad para TREE -> BURNING

# BURNING state transitionsMMM
P_B_B = 0.80  # Probabilidad BURNING -> BURNING (permanecer quemándose) (AUMENTADO de 0.10)
P_B_X = 0.20  # Probabilidad BURNING -> BURNT (X para quemado) (REDUCIDO de 0.90)

# BURNT state transitions
P_X_E_BASE = 0.00000  # Probabilidad base BURNT -> EMPTY (regeneración) (REDUCIDO DRÁSTICAMENTE de 0.05)
F_X_E_NEIGHBOR = 0.00  # Factor de influencia de vecinos TREE para BURNT -> EMPTY (semillas)
P_X_E_MAX_INFLUENCE = 0.0  # Máxima probabilidad adicional por influencia para BURNT -> EMPTY

# Nuevas variables de clima y suelo
temperature_celsius = 25.0  # Temperatura en grados Celsius (ej: 0-50)
soil_moisture = 0.5         # Humedad del suelo (0 a 1, mayor humedad reduce propagación)

# Factores de influencia para nuevas variables (TREE -> BURNING)
TEMP_BASELINE_C = 20.0      # Temperatura base para cálculo de efecto
TEMP_SENSITIVITY = 0.005    # Aumento de probabilidad de ignición por °C sobre el baseline
SOIL_MOISTURE_SENSITIVITY = 0.7 # Factor de reducción por humedad del suelo (0 a 1, 1 es reducción total)

# Variables de clima (controladas por sliders)
wind_direction = [-10, 5]  # Dirección del viento (componente_y, componente_x)
wind_intensity = 0.9  # Intensidad del viento (0 a 1)
humidity = 0.3  # Factor de humedad (0 a 1, mayor humedad reduce propagación)
SIMULATION_SPEED_MULTIPLIER = 1 # Nueva variable global para la velocidad

# --- NUEVO: Pincel de sequedad (0-100) y tamaño ---
brush_size_cells = 1  # tamaño del pincel en celdas (lado del cuadrado)
brush_dryness_value = 50  # sequedad a pintar [0-100]
painting_mode = 'fire'  # 'fire', 'dryness', 'water', 'tree', 'empty'

# --- NUEVO: Parámetros configurables ---
DRYNESS_SPREAD_MULTIPLIER = 0.8  # escala cuánto amplifica la sequedad la propagación
TIME_STEP_MINUTES = 1  # minutos por paso de simulación

# --- NUEVO: Agua ---
water_grid = np.zeros(GRID_SIZE, dtype=np.uint8)  # 1 si hay agua pintada
WATER_EFFECT_RADIUS = 3  # celdas
WATER_SOIL_MOISTURE_BONUS = 0.4  # bonus de humedad del suelo por agua cercana (0-1)
show_water_overlay = False


def get_neighborhood(grid, r, c):
    """
    Obtiene los estados de las 8 celdas vecinas (vecindad de Moore).
    Si un vecino está fuera de la cuadrícula, se considera EMPTY.

    Args:
        grid (np.array): La cuadrícula actual del bosque.
        r (int): Fila de la celda central.
        c (int): Columna de la celda central.

    Returns:
        list: Una lista con los estados de las 8 celdas vecinas.
              El orden es: (-1,-1),(-1,0),(-1,1), (0,-1),(0,1), (1,-1),(1,0),(1,1)
              relativo a la celda (r,c), donde dr es el primer elemento del par.
    """
    neighbors = []
    for dr in [-1, 0, 1]:  # Delta fila
        for dc in [-1, 0, 1]:  # Delta columna
            if dr == 0 and dc == 0:  # Excluimos la celda central
                continue
            nr, nc = r + dr, c + dc
            # Manejo de bordes
            if nr < 0 or nc < 0 or nr >= grid.shape[0] or nc >= grid.shape[1]:
                neighbors.append(EMPTY)  # Consideramos borde como vacío
            else:
                neighbors.append(grid[nr, nc])
    return neighbors

# Mapeo explícito del índice del vecino en la lista `neighborhood` a sus coordenadas (dr, dc)
# Este orden debe coincidir con cómo `get_neighborhood` construye la lista.
NEIGHBOR_RELATIVE_COORDS = [
    (-1, -1), (-1, 0), (-1, 1),  # dr = -1
    (0, -1),           (0, 1),   # dr = 0 (centro excluido)
    (1, -1), (1, 0), (1, 1)   # dr = 1
]

def get_transition_matrix(neighborhood_states, wind_dir, wind_int, hum, temp_c, soil_moist, current_state, local_dryness_0_to_100=0.0):
    """
    Calcula las probabilidades de transición para una celda, dado su estado actual
    y el de sus vecinos, además de las condiciones climáticas.

    Args:
        neighborhood_states (list): Estados de las celdas vecinas.
        wind_dir (list): Dirección del viento [componente_y, componente_x].
        wind_int (float): Intensidad del viento.
        hum (float): Humedad del aire.
        temp_c (float): Temperatura en Celsius.
        soil_moist (float): Humedad del suelo.
        current_state (int): Estado actual de la celda.

    Returns:
        np.array: Vector de probabilidades de transición para el estado actual.
    """
    # Matriz de transición base con valores por defecto
    # Filas: EMPTY, TREE, BURNING, BURNT
    # Columnas: [EMPTY, TREE, BURNING, BURNT]
    base_transitions = np.array([
        [1.0 - P_E_T_BASE, P_E_T_BASE, 0.00, 0.00],  # EMPTY
        [0.00, 1.0 - P_T_B_BASE, P_T_B_BASE, 0.00],  # TREE
        [0.00, 0.00, P_B_B, P_B_X],                 # BURNING ¿Podría apagarse? 
        [P_X_E_BASE, 0.00, 0.00, 1.0 - P_X_E_BASE]  # BURNT -> Llamado: "X" (para no repetir la letra "B")
    ])
    
    probs = base_transitions[current_state].copy() # Copiamos la fila correspondiente al estado actual
    
    burning_neighbors_count = neighborhood_states.count(BURNING)
    tree_neighbors_count = neighborhood_states.count(TREE)
    
    # 1. Transiciones desde EMPTY
    if current_state == EMPTY:
        tree_influence = min(F_E_T_NEIGHBOR * tree_neighbors_count, P_E_T_MAX_INFLUENCE)
        probs[TREE] = P_E_T_BASE + tree_influence
        probs[EMPTY] = 1.0 - probs[TREE]  # Normalizar
    
    # 2. Transiciones desde TREE
    elif current_state == TREE:
        total_wind_factor = 0.0
        if burning_neighbors_count > 0: # El viento solo importa si hay vecinos quemándose
            wind_y_comp, wind_x_comp = wind_dir[0], wind_dir[1]
            norm_wind = math.sqrt(wind_x_comp**2 + wind_y_comp**2)

            for i, neighbor_state in enumerate(neighborhood_states):
                if neighbor_state == BURNING:
                    # dr_n, dc_n son las coordenadas relativas del vecino (desde la celda actual hacia el vecino)
                    dr_n, dc_n = NEIGHBOR_RELATIVE_COORDS[i]
                    
                    # Vector desde el vecino quemándose hacia la celda actual es (-dr_n, -dc_n)
                    # Normalizamos este vector (si no es cero)
                    norm_neighbor_vec = math.sqrt(dr_n**2 + dc_n**2)
                    
                    if norm_neighbor_vec > 0 and norm_wind > 0:
                        # Producto escalar entre (-dr_n, -dc_n) y (wind_x, wind_y)
                        dot_product = (-dr_n * wind_y_comp) + (-dc_n * wind_x_comp) # Note: dr_n is y-like, dc_n is x-like for wind_dir
                                                                                # wind_y_comp, wind_x_comp
                                                                                # dr_n, dc_n from NEIGHBOR_RELATIVE_COORDS
                                                                                # If wind_dir is (y,x) and relative_coords is (dr,dc)
                                                                                # then for standard cartesian dot product:
                                                                                # vec_to_cell_x = -dc_n, vec_to_cell_y = -dr_n
                                                                                # wind_x = wind_dir[1], wind_y = wind_dir[0]
                                                                                # dot_product = (-dc_n * wind_dir[1]) + (-dr_n * wind_dir[0])
                        
                        alignment = dot_product / (norm_neighbor_vec * norm_wind) # alignment está en [-1, 1]
                        # alignment = 1 significa que el viento sopla directamente DESDE el vecino quemándose HACIA la celda.
                        total_wind_factor += alignment # Sumamos la alineación; puede ser positivo o negativo
            
        # Ajustamos por factor de humedad (reduce propagación)
        humidity_reduction_factor = 1.0 - hum
        
        # Efecto de la temperatura
        temp_effect_on_spread = 0.0 # Initialize
        # El efecto de la temperatura ahora solo contribuye si hay vecinos quemándose,
        # mejorando la propagación en lugar de causar nuevas igniciones aisladas.
        if burning_neighbors_count > 0:
            if temp_c > TEMP_BASELINE_C:
                temp_effect_on_spread = (temp_c - TEMP_BASELINE_C) * TEMP_SENSITIVITY
        
        # Efecto de la humedad del suelo (reduce propagación)
        soil_moisture_reduction_factor = 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY)
        soil_moisture_reduction_factor = max(0.0, soil_moisture_reduction_factor) # Asegurar que no sea negativo
        
        # Probabilidad de quemarse: base + influencia de vecinos + efecto del viento (ajustado por humedad)
        # El factor de vecinos quemándose y el del viento se aplican sobre la susceptibilidad reducida por la humedad.
        ignition_prob_from_neighbors = F_T_B_NEIGHBOR * burning_neighbors_count
        ignition_prob_from_wind = total_wind_factor * wind_int * F_T_B_WIND # wind_int [0,1]
        
        # La humedad afecta la propagación general de los vecinos y el viento
        # Se añade el efecto de la temperatura y se multiplica por el factor de reducción de humedad del suelo
        # NUEVO: efecto de sequedad local [0-100] como amplificador de propagación
        dryness_scale = 1.0 + (max(0.0, min(100.0, float(local_dryness_0_to_100))) / 100.0) * DRYNESS_SPREAD_MULTIPLIER
        effective_spread_prob = (ignition_prob_from_neighbors + ignition_prob_from_wind + temp_effect_on_spread) \
                                * humidity_reduction_factor * soil_moisture_reduction_factor * dryness_scale
        
        final_burning_prob = P_T_B_BASE + effective_spread_prob
        final_burning_prob = min(max(final_burning_prob, 0.0), P_T_B_MAX) # Asegurar que esté entre 0 y P_T_B_MAX

        probs[BURNING] = final_burning_prob
        probs[TREE] = 1.0 - final_burning_prob # Normalizar
        if probs[TREE] < 0: probs[TREE] = 0 # Evitar probabilidad negativa por si acaso
        probs[EMPTY] = 0 # Un arbol no se vuelve vacio directamente por fuego

    # 3. Transiciones desde BURNING (ya definidas en base_transitions, no se modifican por ahora)
    elif current_state == BURNING:
        pass # Mantiene P_B_B y P_B_X
        
    # 4. Transiciones desde BURNT
    elif current_state == BURNT:
        regen_influence = min(F_X_E_NEIGHBOR * tree_neighbors_count, P_X_E_MAX_INFLUENCE)
        probs[EMPTY] = P_X_E_BASE + regen_influence
        probs[BURNT] = 1.0 - probs[EMPTY] # Normalizar

    # Asegurar que la suma de probabilidades sea 1 y no haya negativas
    probs = np.maximum(probs, 0) # No negativas
    probs_sum = np.sum(probs)
    if probs_sum == 0: # Evitar división por cero si todos son 0 (improbable pero seguro)
        # Volver a un estado por defecto o no cambiar
        if current_state == EMPTY: probs[EMPTY] = 1.0
        elif current_state == TREE: probs[TREE] = 1.0
        elif current_state == BURNING: probs[BURNING] = 1.0 # O probs[BURNT]=1.0
        elif current_state == BURNT: probs[BURNT] = 1.0
    else:
        probs = probs / probs_sum # Normalizar

    return probs


def pure_markov_fire_model(grid):
    """
    Aplica el modelo de incendio basado en cadenas de Markov para actualizar la cuadrícula.

    Args:
        grid (np.array): La cuadrícula actual del bosque.

    Returns:
        np.array: La nueva cuadrícula después de una iteración.
    """
    new_grid = grid.copy()
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            # Tratamiento de agua como EMPTY
            if water_grid[r, c] > 0:
                current_cell_state = EMPTY
            neighborhood = get_neighborhood(grid, r, c)
            current_cell_state = grid[r, c]
            
            local_dryness = 0.0
            try:
                local_dryness = float(dryness_grid[r, c])
            except Exception:
                local_dryness = 0.0

            # Humedad del suelo local por agua cercana
            r0 = max(0, r - WATER_EFFECT_RADIUS)
            r1 = min(GRID_SIZE[0], r + WATER_EFFECT_RADIUS + 1)
            c0 = max(0, c - WATER_EFFECT_RADIUS)
            c1 = min(GRID_SIZE[1], c + WATER_EFFECT_RADIUS + 1)
            near_water = np.any(water_grid[r0:r1, c0:c1] > 0)
            local_soil = soil_moisture + (WATER_SOIL_MOISTURE_BONUS if near_water else 0.0)
            if local_soil > 1.0:
                local_soil = 1.0
            transition_probs = get_transition_matrix(
                neighborhood, wind_direction, wind_intensity, humidity,
                temperature_celsius, local_soil, # humedad del suelo local ajustada por agua
                current_cell_state,
                local_dryness_0_to_100=local_dryness
            )
            
            # Forzar que el agua permanezca EMPTY
            if water_grid[r, c] > 0:
                new_grid[r, c] = EMPTY
            else:
                new_grid[r, c] = np.random.choice([EMPTY, TREE, BURNING, BURNT], p=transition_probs)
    
    return new_grid

# Inicializa la cuadrícula con árboles
forest = np.random.choice([EMPTY, TREE], size=GRID_SIZE, p=[0.1, 0.9])
# NUEVO: cuadrícula de sequedad [0-100]
dryness_grid = np.full(GRID_SIZE, 50.0, dtype=float)
# Fuego inicial en el centro - Comentado para que no inicie con fuego automáticamente
# forest[GRID_SIZE[0] // 2, GRID_SIZE[1] // 2] = BURNING


def compute_statistics(grid):
    """
    Calcula las estadísticas de los estados de las celdas en la cuadrícula.

    Args:
        grid (np.array): La cuadrícula actual del bosque.

    Returns:
        dict: Un diccionario con el porcentaje de cada tipo de celda.
    """
    total_cells = grid.size
    empty_count = np.sum(grid == EMPTY)
    tree_count = np.sum(grid == TREE)
    burning_count = np.sum(grid == BURNING)
    burnt_count = np.sum(grid == BURNT)
    
    return {
        'empty': empty_count / total_cells * 100,
        'tree': tree_count / total_cells * 100,
        'burning': burning_count / total_cells * 100,
        'burnt': burnt_count / total_cells * 100
    }

# Variables para almacenar estadísticas durante la simulación
stats_history = []

# Definir colores personalizados para cada estado
colors = ["white", "green", "red", "black"]  # Vacío, Árbol, Fuego, Quemado
def build_display_image(grid_states, grid_dryness):
    h, w = grid_states.shape
    img = np.zeros((h, w, 3), dtype=float)
    # Colores base
    color_empty = np.array([1.0, 1.0, 1.0])
    # Verde más oscuro para mayor contraste en bajas sequedades
    color_green = np.array([0.0, 0.5, 0.0])
    color_yellow = np.array([1.0, 1.0, 0.0])
    color_red = np.array([1.0, 0.0, 0.0])
    color_blue = np.array([0.0, 0.4, 1.0])
    color_black = np.array([0.0, 0.0, 0.0])

    # Máscaras por estado
    mask_empty = (grid_states == EMPTY)
    mask_tree = (grid_states == TREE)
    mask_burning = (grid_states == BURNING)
    mask_burnt = (grid_states == BURNT)

    # Empty
    img[mask_empty] = color_empty

    # Tree -> degradado según sequedad (0 -> verde, 100 -> amarillo)
    if grid_dryness is not None:
        dryness_norm = np.clip(grid_dryness / 100.0, 0.0, 1.0)
    else:
        dryness_norm = 0.5
    # Mezcla por celdas TREE
    tree_idx = np.where(mask_tree)
    if tree_idx[0].size > 0:
        # Curva no lineal para resaltar diferencias 0-80
        w_yellow = np.power(dryness_norm[tree_idx], 1.5)
        w_green = 1.0 - w_yellow
        img[tree_idx] = (w_green[:, None] * color_green) + (w_yellow[:, None] * color_yellow)

    # Burning
    img[mask_burning] = color_red
    # Burnt
    img[mask_burnt] = color_black

    # Celdas de agua: azules (override visual) + overlay de humedad con tramado
    if np.any(water_grid):
        water_mask = (water_grid > 0)
        img[water_mask] = color_blue
        if show_water_overlay:
            pad = WATER_EFFECT_RADIUS
            padded = np.pad(water_grid, pad_width=pad, mode='constant', constant_values=0)
            near = np.zeros_like(water_grid, dtype=bool)
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    near |= padded[pad + dr:pad + dr + water_grid.shape[0], pad + dc:pad + dc + water_grid.shape[1]] > 0
            overlay_mask = near & (~water_mask)
            if np.any(overlay_mask):
                cyan = np.array([0.6, 0.9, 1.0])
                alpha = 0.25
                img[overlay_mask] = (1 - alpha) * img[overlay_mask] + alpha * cyan
                # Tramado simple: rayas diagonales cada 2 celdas
                rr, cc = np.where(overlay_mask)
                stripes = ((rr + cc) % 4 == 0)
                if np.any(stripes):
                    idx = (rr[stripes], cc[stripes])
                    img[idx] = (1 - 0.35) * img[idx] + 0.35 * np.array([0.5, 0.85, 1.0])

    return img

# Configuración de la animación
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8)) # Ajustado tamaño figura
im = ax1.imshow(build_display_image(forest, dryness_grid), animated=True)
ax1.set_title("Simulación de Incendio (Autómata Celular Estocástico)")
ax1.set_xticks([]) # Ocultar ticks del eje X
ax1.set_yticks([]) # Ocultar ticks del eje Y

# Para el gráfico de estadísticas
# Cambiado el color de 'Vacío' a 'lightgray' para mejor visibilidad
line_empty, = ax2.plot([], [], color='lightgray', linestyle='-', label='Vacío (%)') 
line_tree, = ax2.plot([], [], color=colors[TREE], linestyle='-', label='Árbol (%)')
line_burning, = ax2.plot([], [], color=colors[BURNING], linestyle='-', label='Quemando (%)')
line_burnt, = ax2.plot([], [], color=colors[BURNT], linestyle='-', label='Quemado (%)')
ax2.set_xlim(0, SIMULATION_STEPS) # Se ajustará dinámicamente
ax2.set_ylim(0, 100)
ax2.set_xlabel('Pasos de Simulación Reales')
ax2.set_ylabel('Porcentaje de Celdas (%)')
ax2.legend()
ax2.set_title("Evolución de Estados del Ecosistema")
ax2.grid(True)

# Ajustar el layout para hacer espacio a los sliders y botón
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.45) # Más espacio abajo para sliders

# --- Variables y funciones para el inicio interactivo del fuego y pincel ---
simulation_started = False
simulation_paused = False
ani = None
current_highlight_patch = None
is_mouse_button_down = False # NUEVA: para rastrear si el botón del ratón está presionado
brush_info_text = None

def update_brush_text():
    global brush_info_text
    status = f"Modo: {painting_mode.upper()} | Pincel: {brush_size_cells} | Sequedad: {int(brush_dryness_value)}"
    if brush_info_text is None:
        brush_info_text = fig.text(0.05, 0.94, status)
    else:
        brush_info_text.set_text(status)
    fig.canvas.draw_idle()

def apply_dryness_brush(r_center, c_center):
    global dryness_grid
    half = brush_size_cells // 2
    r_start = max(0, r_center - half)
    r_end = min(GRID_SIZE[0], r_start + brush_size_cells)
    c_start = max(0, c_center - half)
    c_end = min(GRID_SIZE[1], c_start + brush_size_cells)
    dryness_grid[r_start:r_end, c_start:c_end] = float(brush_dryness_value)

def apply_state_brush(r_center, c_center, state_value):
    global forest
    half = brush_size_cells // 2
    r_start = max(0, r_center - half)
    r_end = min(GRID_SIZE[0], r_start + brush_size_cells)
    c_start = max(0, c_center - half)
    c_end = min(GRID_SIZE[1], c_start + brush_size_cells)
    forest[r_start:r_end, c_start:c_end] = state_value

def apply_water_brush(r_center, c_center):
    global water_grid
    half = brush_size_cells // 2
    r_start = max(0, r_center - half)
    r_end = min(GRID_SIZE[0], r_start + brush_size_cells)
    c_start = max(0, c_center - half)
    c_end = min(GRID_SIZE[1], c_start + brush_size_cells)
    water_grid[r_start:r_end, c_start:c_end] = 1

def on_click_fire_start(event):
    print("--- on_click_fire_start LLAMADA ---")
    global forest, im, is_mouse_button_down # Añadido is_mouse_button_down
    
    if event.inaxes != ax1:
        # print("Clic fuera de ax1") # Puede ser ruidoso, opcional
        return
    
    if event.button == 1: # Botón izquierdo del ratón
        is_mouse_button_down = True # Marcar que el botón está presionado
        # Aplicar acción en la celda del clic inicial
        c, r = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r < GRID_SIZE[0] and 0 <= c < GRID_SIZE[1]:
            if painting_mode == 'fire':
                apply_state_brush(r, c, BURNING)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Fuego (clic inicial) en: Fila {r}, Columna {c}")
            elif painting_mode == 'dryness':
                apply_dryness_brush(r, c)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Sequedad={int(brush_dryness_value)} (clic inicial) en: Fila {r}, Columna {c}, tamaño={brush_size_cells}")
            elif painting_mode == 'water':
                apply_water_brush(r, c)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Agua (clic inicial) en: Fila {r}, Columna {c}, tamaño={brush_size_cells}")
            elif painting_mode == 'tree':
                apply_state_brush(r, c, TREE)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Árbol (clic inicial) en: Fila {r}, Columna {c}")
            elif painting_mode == 'empty':
                apply_state_brush(r, c, EMPTY)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Vacío (clic inicial) en: Fila {r}, Columna {c}")
        else:
            print("Clic inicial fuera de la cuadrícula.")
    # No es necesario llamar a fig.canvas.draw_idle() aquí si on_hover lo va a hacer

def on_release_fire(event):
    global is_mouse_button_down
    if event.button == 1: # Botón izquierdo del ratón
        is_mouse_button_down = False # Marcar que el botón se soltó
        print("--- Botón del ratón SOLTADO ---")

fig.canvas.mpl_connect('button_press_event', on_click_fire_start)
fig.canvas.mpl_connect('button_release_event', on_release_fire) # NUEVA CONEXIÓN

def on_hover(event):
    global current_highlight_patch, forest, im, is_mouse_button_down # Añadido forest, im, is_mouse_button_down
    
    # Primero, manejar el resaltado
    if event.inaxes == ax1:
        c_hover, r_hover = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r_hover < GRID_SIZE[0] and 0 <= c_hover < GRID_SIZE[1]:
            if current_highlight_patch:
                current_highlight_patch.remove()
            current_highlight_patch = patches.Rectangle(
                (c_hover - 0.5, r_hover - 0.5), 1, 1, linewidth=1.5, edgecolor='yellow', facecolor='none', zorder=10
            )
            ax1.add_patch(current_highlight_patch)
            # No llamar a draw_idle() aquí solo por el hover si estamos pintando, para evitar demasiados redibujados.
        else: # Fuera de la cuadrícula, pero el ratón aún podría estar presionado
            if current_highlight_patch:
                current_highlight_patch.remove()
                current_highlight_patch = None
                # fig.canvas.draw_idle() # Solo redibujar si se quitó el resaltado
    else: # Fuera de ax1
        if current_highlight_patch:
            current_highlight_patch.remove()
            current_highlight_patch = None
            # fig.canvas.draw_idle()

    # Luego, manejar el pintado si el botón está presionado
    if is_mouse_button_down and event.inaxes == ax1:
        c_paint, r_paint = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r_paint < GRID_SIZE[0] and 0 <= c_paint < GRID_SIZE[1]:
            if painting_mode == 'fire':
                apply_state_brush(r_paint, c_paint, BURNING)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Pintando fuego en: Fila {r_paint}, Columna {c_paint}")
            elif painting_mode == 'dryness':
                apply_dryness_brush(r_paint, c_paint)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Pintando sequedad={int(brush_dryness_value)} en: Fila {r_paint}, Columna {c_paint}")
            elif painting_mode == 'water':
                apply_water_brush(r_paint, c_paint)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Pintando agua en: Fila {r_paint}, Columna {c_paint}")
            elif painting_mode == 'tree':
                apply_state_brush(r_paint, c_paint, TREE)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Pintando árbol en: Fila {r_paint}, Columna {c_paint}")
            elif painting_mode == 'empty':
                apply_state_brush(r_paint, c_paint, EMPTY)
                im.set_array(build_display_image(forest, dryness_grid))
                print(f"Pintando vacío en: Fila {r_paint}, Columna {c_paint}")
    
    # Un solo draw_idle al final de on_hover si algo cambió o es necesario
    # Esto es un poco heurístico; demasiados draw_idle pueden alentar.
    # Si is_mouse_button_down, es probable que queramos ver el pintado.
    if (event.inaxes == ax1 and (is_mouse_button_down or current_highlight_patch is not None)) or \
       (event.inaxes != ax1 and current_highlight_patch is None): # Condición para redibujar si salimos de ax1 y se quitó el patch
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)

def on_key(event):
    global painting_mode, brush_size_cells, brush_dryness_value, show_water_overlay
    if event.key == 'm':
        # Rotación fire -> dryness -> water -> tree -> empty -> fire
        if painting_mode == 'fire':
            painting_mode = 'dryness'
        elif painting_mode == 'dryness':
            painting_mode = 'water'
        elif painting_mode == 'water':
            painting_mode = 'tree'
        elif painting_mode == 'tree':
            painting_mode = 'empty'
        else:
            painting_mode = 'fire'
        update_brush_text()
    elif event.key in ['.']:
        brush_size_cells = min(25, brush_size_cells + 1)
        update_brush_text()
    elif event.key == ',':
        brush_size_cells = max(1, brush_size_cells - 1)
        update_brush_text()
    elif event.key == '|':
        brush_dryness_value = 0
        update_brush_text()  
    elif event.key == '1':
        brush_dryness_value = 10
        update_brush_text()
    elif event.key == '2':
        brush_dryness_value = 20
        update_brush_text()
    elif event.key == '3':
        brush_dryness_value = 30
        update_brush_text()
    elif event.key == '4':
        brush_dryness_value = 40
        update_brush_text()
    elif event.key == '5':
        brush_dryness_value = 50
        update_brush_text()
    elif event.key == '6':
        brush_dryness_value = 60
        update_brush_text()
    elif event.key == '7':
        brush_dryness_value = 70
        update_brush_text()
    elif event.key == '8':
        brush_dryness_value = 80
        update_brush_text()
    elif event.key == '9':
        brush_dryness_value = 90
        update_brush_text() 
    elif event.key == '0':
        brush_dryness_value = 100
        update_brush_text()
    elif event.key == '{':
        brush_dryness_value = max(0, brush_dryness_value - 5)
        update_brush_text()
    elif event.key == '}':
        brush_dryness_value = min(100, brush_dryness_value + 5)
        update_brush_text()
    elif event.key == 'h':
        show_water_overlay = not show_water_overlay
        im.set_array(build_display_image(forest, dryness_grid))
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

# Crear ejes para los sliders
# Ajuste de posiciones para acomodar los nuevos sliders
ax_wind_x = plt.axes([0.15, 0.40, 0.7, 0.025])
ax_wind_y = plt.axes([0.15, 0.35, 0.7, 0.025])
ax_intensity = plt.axes([0.15, 0.30, 0.7, 0.025])
ax_humidity = plt.axes([0.15, 0.25, 0.7, 0.025]) # Este es Humedad (aire)
ax_temperature = plt.axes([0.15, 0.20, 0.7, 0.025]) # Nuevo para Temperatura
ax_soil_moisture = plt.axes([0.15, 0.15, 0.7, 0.025]) # Nuevo para Humedad Suelo
ax_speed = plt.axes([0.15, 0.10, 0.7, 0.025]) # Slider de velocidad
ax_brush_size = plt.axes([0.15, 0.075, 0.32, 0.025])
ax_brush_dryness = plt.axes([0.53, 0.075, 0.32, 0.025])
ax_dryness_multiplier = plt.axes([0.15, 0.05, 0.32, 0.02])

# Crear sliders
slider_wind_x = Slider(ax_wind_x, 'Viento X', -10.0, 10.0, valinit=wind_direction[1], valstep=0.5)
slider_wind_y = Slider(ax_wind_y, 'Viento Y', -10.0, 10.0, valinit=wind_direction[0], valstep=0.5)
slider_intensity = Slider(ax_intensity, 'Intensidad Viento', 0.0, 1.0, valinit=wind_intensity, valstep=0.05)
slider_humidity = Slider(ax_humidity, 'Humedad Aire', 0.0, 1.0, valinit=humidity, valstep=0.05) # Etiqueta actualizada
slider_temperature = Slider(ax_temperature, 'Temperatura (°C)', 0.0, 50.0, valinit=temperature_celsius, valstep=1)
slider_soil_moisture = Slider(ax_soil_moisture, 'Humedad Suelo', 0.0, 1.0, valinit=soil_moisture, valstep=0.05)
slider_speed = Slider(ax_speed, 'Velocidad (Pasos/Cuadro)', 1, 10, valinit=SIMULATION_SPEED_MULTIPLIER, valstep=1)
slider_brush_size = Slider(ax_brush_size, 'Tamaño Pincel (celdas)', 1, 25, valinit=brush_size_cells, valstep=1)
slider_brush_dryness = Slider(ax_brush_dryness, 'Sequedad [0-100]', 0, 100, valinit=brush_dryness_value, valstep=1)
slider_dryness_multiplier = Slider(ax_dryness_multiplier, 'Multiplicador Sequedad', 0.0, 2.0, valinit=DRYNESS_SPREAD_MULTIPLIER, valstep=0.05)

# Funciones para actualizar las variables globales cuando los sliders cambian
def update_wind_x(val):
    global wind_direction
    wind_direction[1] = val

def update_wind_y(val):
    global wind_direction
    wind_direction[0] = val

def update_intensity(val):
    global wind_intensity
    wind_intensity = val

def update_humidity(val):
    global humidity
    humidity = val

def update_temperature(val):
    global temperature_celsius
    temperature_celsius = val

def update_soil_moisture(val):
    global soil_moisture
    soil_moisture = val

def update_speed(val):
    global SIMULATION_SPEED_MULTIPLIER
    SIMULATION_SPEED_MULTIPLIER = int(val)

def update_brush_size(val):
    global brush_size_cells
    brush_size_cells = int(val)
    update_brush_text()

def update_brush_dryness(val):
    global brush_dryness_value
    brush_dryness_value = int(val)
    update_brush_text()

def update_dryness_multiplier(val):
    global DRYNESS_SPREAD_MULTIPLIER
    DRYNESS_SPREAD_MULTIPLIER = float(val)

def set_time_step(minutes):
    # función obsoleta, reservada si más tarde se desea reintroducir UI
    return

# Conectar sliders a las funciones de actualización
slider_wind_x.on_changed(update_wind_x)
slider_wind_y.on_changed(update_wind_y)
slider_intensity.on_changed(update_intensity)
slider_humidity.on_changed(update_humidity)
slider_temperature.on_changed(update_temperature) # Conectar nuevo slider
slider_soil_moisture.on_changed(update_soil_moisture) # Conectar nuevo slider
slider_speed.on_changed(update_speed)
slider_brush_size.on_changed(update_brush_size)
slider_brush_dryness.on_changed(update_brush_dryness)
slider_dryness_multiplier.on_changed(update_dryness_multiplier)

# --- Botones de Control --- #
DEFAULT_BUTTON_COLOR = 'lightgoldenrodyellow' # Color por defecto para botones

# Botón Principal (Iniciar/Pausar/Reanudar)
ax_button_main = plt.axes([0.25, 0.02, 0.2, 0.035]) # Posición ajustada para bottom=0.45
button_main = Button(ax_button_main, 'Iniciar Simulación', color=DEFAULT_BUTTON_COLOR, hovercolor='0.975')

# Botón Finalizar (inicialmente oculto)
ax_button_finalize = plt.axes([0.55, 0.02, 0.2, 0.035]) # Posición ajustada para bottom=0.45
button_finalize = Button(ax_button_finalize, 'Finalizar Simulación', color=DEFAULT_BUTTON_COLOR, hovercolor='0.975')
button_finalize.ax.set_visible(False) # Oculto al inicio

def reset_simulation_state():
    """Resetea la simulación a su estado inicial para una nueva ejecución."""
    global forest, stats_history, im, line_empty, line_tree, line_burning, line_burnt, simulation_started, simulation_paused
    
    simulation_started = False
    simulation_paused = False
    
    forest = np.random.choice([EMPTY, TREE], size=GRID_SIZE, p=[0.1, 0.9])
    # Reiniciar sequedad
    global dryness_grid, water_grid
    dryness_grid = np.full(GRID_SIZE, 50.0, dtype=float)
    water_grid = np.zeros(GRID_SIZE, dtype=np.uint8)
    im.set_array(build_display_image(forest, dryness_grid))
    
    stats_history = []
    x_data = []
    empty_data = []
    tree_data = []
    burning_data = []
    burnt_data = []
    
    line_empty.set_data(x_data, empty_data)
    line_tree.set_data(x_data, tree_data)
    line_burning.set_data(x_data, burning_data)
    line_burnt.set_data(x_data, burnt_data)
    ax2.set_xlim(0, SIMULATION_STEPS) # Resetear xlim del gráfico de stats
    # ax2.relim()
    # ax2.autoscale_view()
    
    button_main.label.set_text("Iniciar Simulación")
    if hasattr(button_main, 'ax'): button_main.ax.set_facecolor(DEFAULT_BUTTON_COLOR)
    button_finalize.ax.set_visible(False)
    
    print("Simulación reseteada. Lista para un nuevo inicio.")
    fig.canvas.draw_idle()
    update_brush_text()

def main_button_callback(event):
    global simulation_started, simulation_paused, ani, forest, stats_history

    if not simulation_started:
        # CASO: Iniciar Simulación
        print("Iniciando simulación...")
        simulation_started = True
        simulation_paused = False # Asegurarse que no está pausada al iniciar
        
        button_main.label.set_text("Pausar")
        if hasattr(button_main, 'ax'): button_main.ax.set_facecolor('red')
        button_finalize.ax.set_visible(False) # Asegurarse que Finalizar está oculto

        if BURNING not in forest:
            print("No se seleccionó punto de inicio. Iniciando fuego en el centro.")
            forest[GRID_SIZE[0] // 2, GRID_SIZE[1] // 2] = BURNING
            im.set_array(build_display_image(forest, dryness_grid))
        
        stats_history = [] # Limpiar historial por si acaso
        # Crear la animación AHORA, con blit=False y el intervalo fijo de 100ms
        ani = animation.FuncAnimation(fig, update_animation, frames=SIMULATION_STEPS, interval=100, blit=False) 
        fig.canvas.draw_idle()
        update_brush_text()
    
    elif simulation_started and not simulation_paused:
        # CASO: Pausar Simulación
        print("Simulación PAUSADA.")
        simulation_paused = True
        if ani: ani.event_source.stop()
        button_main.label.set_text("Reanudar")
        if hasattr(button_main, 'ax'): button_main.ax.set_facecolor('lightgreen')
        button_finalize.ax.set_visible(True)
        fig.canvas.draw_idle()

    elif simulation_started and simulation_paused:
        # CASO: Reanudar Simulación
        print("Simulación REANUDADA.")
        simulation_paused = False
        if ani: ani.event_source.start()
        button_main.label.set_text("Pausar")
        if hasattr(button_main, 'ax'): button_main.ax.set_facecolor('red')
        button_finalize.ax.set_visible(False)
        fig.canvas.draw_idle()

button_main.on_clicked(main_button_callback)

def finalize_button_callback(event):
    global ani, simulation_started
    if not simulation_started or not simulation_paused: # Solo finalizar si está iniciada y pausada
        print("Comando Finalizar no aplicable en este estado.")
        return

    print("Simulación FINALIZADA por el usuario.")
    if ani:
        ani.event_source.stop() # Detener la animación completamente
        # ani = None # Opcional: destruir referencia a la animación
    reset_simulation_state()

button_finalize.on_clicked(finalize_button_callback)

def update_animation(frame):
    global forest, stats_history, simulation_started, simulation_paused, im, SIMULATION_SPEED_MULTIPLIER
    global line_empty, line_tree, line_burning, line_burnt

    if not simulation_started or simulation_paused:
        # No hacer nada si no ha iniciado o está en pausa.
        # El print de "PREMATURAMENTE" puede ser ruidoso si la pausa funciona bien.
        # print(f"Update_animation skip: started={simulation_started}, paused={simulation_paused}")
        return [im, line_empty, line_tree, line_burning, line_burnt]

    # ... (resto de la lógica de update_animation como estaba)
    for _ in range(SIMULATION_SPEED_MULTIPLIER):
        forest = pure_markov_fire_model(forest)
        current_stats = compute_statistics(forest)
        stats_history.append(current_stats)
    
    im.set_array(build_display_image(forest, dryness_grid))
    
    x_data = list(range(len(stats_history)))
    line_empty.set_data(x_data, [s['empty'] for s in stats_history])
    line_tree.set_data(x_data, [s['tree'] for s in stats_history])
    line_burning.set_data(x_data, [s['burning'] for s in stats_history])
    line_burnt.set_data(x_data, [s['burnt'] for s in stats_history])
    
    if x_data:
        ax2.set_xlim(0, max(SIMULATION_STEPS * SIMULATION_SPEED_MULTIPLIER, len(x_data)) + 1)

    return [im, line_empty, line_tree, line_burning, line_burnt]

# Conectar botones de tiempo
pass

# La animación (ani) NO se crea aquí, sino en start_simulation_callback
# ani = animation.FuncAnimation(fig, update_animation, frames=SIMULATION_STEPS, interval=300, blit=True) # ESTA LÍNEA DEBE ESTAR COMENTADA O ELIMINADA
plt.show() 