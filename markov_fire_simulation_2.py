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
P_E_T_BASE = 0.00001  # Probabilidad base EMPTY -> TREE (REDUCIDO DRÁSTICAMENTE de 0.02)
F_E_T_NEIGHBOR = 0.05  # Factor de influencia de vecinos TREE para EMPTY -> TREE 
P_E_T_MAX_INFLUENCE = 0.02  # Máxima probabilidad adicional por influencia para EMPTY -> TREE (REDUCIDO de 0.3)

# TREE state transitions
P_T_B_BASE = 0.00000  # Probabilidad base TREE -> BURNING (REDUCIDO DRÁSTICAMENTE de 0.05)
F_T_B_NEIGHBOR = 0.15  # Factor de propagación por vecinos BURNING para TREE -> BURNING
F_T_B_WIND = 0.35  # Factor de escala para la influencia del viento en TREE -> BURNING (AUMENTADO de 0.1)
P_T_B_MAX = 0.95  # Máxima probabilidad para TREE -> BURNING

# BURNING state transitions
P_B_B = 0.80  # Probabilidad BURNING -> BURNING (permanecer quemándose) (AUMENTADO de 0.10)
P_B_X = 0.20  # Probabilidad BURNING -> BURNT (X para quemado) (REDUCIDO de 0.90)

# BURNT state transitions
P_X_E_BASE = 0.00001  # Probabilidad base BURNT -> EMPTY (regeneración) (REDUCIDO DRÁSTICAMENTE de 0.05)
F_X_E_NEIGHBOR = 0.02  # Factor de influencia de vecinos TREE para BURNT -> EMPTY (semillas)
P_X_E_MAX_INFLUENCE = 0.2  # Máxima probabilidad adicional por influencia para BURNT -> EMPTY

# Variables de clima (controladas por sliders)
wind_direction = [-10, 5]  # Dirección del viento (componente_y, componente_x)
wind_intensity = 0.9  # Intensidad del viento (0 a 1)
humidity = 0.3  # Factor de humedad (0 a 1, mayor humedad reduce propagación)
SIMULATION_SPEED_MULTIPLIER = 1 # Nueva variable global para la velocidad


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

def get_transition_matrix(neighborhood_states, wind_dir, wind_int, hum, current_state):
    """
    Calcula las probabilidades de transición para una celda, dado su estado actual
    y el de sus vecinos, además de las condiciones climáticas.

    Args:
        neighborhood_states (list): Estados de las celdas vecinas.
        wind_dir (list): Dirección del viento [componente_y, componente_x].
        wind_int (float): Intensidad del viento.
        hum (float): Humedad.
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
        [P_X_E_BASE, 0.00, 0.00, 1.0 - P_X_E_BASE]  # BURNT
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
        
        # Probabilidad de quemarse: base + influencia de vecinos + efecto del viento (ajustado por humedad)
        # El factor de vecinos quemándose y el del viento se aplican sobre la susceptibilidad reducida por la humedad.
        ignition_prob_from_neighbors = F_T_B_NEIGHBOR * burning_neighbors_count
        ignition_prob_from_wind = total_wind_factor * wind_int * F_T_B_WIND # wind_int [0,1]
        
        # La humedad afecta la propagación general de los vecinos y el viento
        effective_spread_prob = (ignition_prob_from_neighbors + ignition_prob_from_wind) * humidity_reduction_factor
        
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
            neighborhood = get_neighborhood(grid, r, c)
            current_cell_state = grid[r, c]
            
            transition_probs = get_transition_matrix(
                neighborhood, wind_direction, wind_intensity, humidity, current_cell_state
            )
            
            new_grid[r, c] = np.random.choice([EMPTY, TREE, BURNING, BURNT], p=transition_probs)
    
    return new_grid

# Inicializa la cuadrícula con árboles
forest = np.random.choice([EMPTY, TREE], size=GRID_SIZE, p=[0.1, 0.9])
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
cmap = ListedColormap(colors)

# Configuración de la animación
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8)) # Ajustado tamaño figura
im = ax1.imshow(forest, cmap=cmap, vmin=0, vmax=3, animated=True)
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
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.35) # Más espacio abajo

# --- Variables y funciones para el inicio interactivo del fuego ---
simulation_started = False
simulation_paused = False
ani = None
current_highlight_patch = None
is_mouse_button_down = False # NUEVA: para rastrear si el botón del ratón está presionado

def on_click_fire_start(event):
    print("--- on_click_fire_start LLAMADA ---")
    global forest, im, is_mouse_button_down # Añadido is_mouse_button_down
    
    if event.inaxes != ax1:
        # print("Clic fuera de ax1") # Puede ser ruidoso, opcional
        return
    
    if event.button == 1: # Botón izquierdo del ratón
        is_mouse_button_down = True # Marcar que el botón está presionado
        # Aplicar fuego en la celda actual del clic inicial
        c, r = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r < GRID_SIZE[0] and 0 <= c < GRID_SIZE[1]:
            forest[r, c] = BURNING
            im.set_array(forest)
            # fig.canvas.draw_idle() # on_hover se encargará del redibujado si se mueve
            print(f"Fuego (clic inicial) en: Fila {r}, Columna {c}")
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

    # Luego, manejar el "pintar con fuego" si el botón está presionado
    if is_mouse_button_down and event.inaxes == ax1:
        c_paint, r_paint = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r_paint < GRID_SIZE[0] and 0 <= c_paint < GRID_SIZE[1]:
            if forest[r_paint, c_paint] != BURNING: # Solo cambiar si no está ya quemándose
                forest[r_paint, c_paint] = BURNING
                im.set_array(forest)
                print(f"Pintando fuego en: Fila {r_paint}, Columna {c_paint}")
    
    # Un solo draw_idle al final de on_hover si algo cambió o es necesario
    # Esto es un poco heurístico; demasiados draw_idle pueden alentar.
    # Si is_mouse_button_down, es probable que queramos ver el pintado.
    if (event.inaxes == ax1 and (is_mouse_button_down or current_highlight_patch is not None)) or \
       (event.inaxes != ax1 and current_highlight_patch is None): # Condición para redibujar si salimos de ax1 y se quitó el patch
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Crear ejes para los sliders
ax_wind_x = plt.axes([0.15, 0.25, 0.7, 0.025]) # Ligeramente subidos y ajustados
ax_wind_y = plt.axes([0.15, 0.20, 0.7, 0.025])
ax_intensity = plt.axes([0.15, 0.15, 0.7, 0.025])
ax_humidity = plt.axes([0.15, 0.10, 0.7, 0.025])
ax_speed = plt.axes([0.15, 0.05, 0.7, 0.025]) # Nuevo slider para velocidad

# Crear sliders
slider_wind_x = Slider(ax_wind_x, 'Viento X', -10.0, 10.0, valinit=wind_direction[1], valstep=0.5)
slider_wind_y = Slider(ax_wind_y, 'Viento Y', -10.0, 10.0, valinit=wind_direction[0], valstep=0.5)
slider_intensity = Slider(ax_intensity, 'Intensidad Viento', 0.0, 1.0, valinit=wind_intensity, valstep=0.05)
slider_humidity = Slider(ax_humidity, 'Humedad', 0.0, 1.0, valinit=humidity, valstep=0.05)
slider_speed = Slider(ax_speed, 'Velocidad (Pasos/Cuadro)', 1, 10, valinit=SIMULATION_SPEED_MULTIPLIER, valstep=1)

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

def update_speed(val):
    global SIMULATION_SPEED_MULTIPLIER
    SIMULATION_SPEED_MULTIPLIER = int(val)

# Conectar sliders a las funciones de actualización
slider_wind_x.on_changed(update_wind_x)
slider_wind_y.on_changed(update_wind_y)
slider_intensity.on_changed(update_intensity)
slider_humidity.on_changed(update_humidity)
slider_speed.on_changed(update_speed)

# --- Botones de Control --- #
DEFAULT_BUTTON_COLOR = 'lightgoldenrodyellow' # Color por defecto para botones

# Botón Principal (Iniciar/Pausar/Reanudar)
ax_button_main = plt.axes([0.25, 0.005, 0.2, 0.035]) # Posición ajustada
button_main = Button(ax_button_main, 'Iniciar Simulación', color=DEFAULT_BUTTON_COLOR, hovercolor='0.975')

# Botón Finalizar (inicialmente oculto)
ax_button_finalize = plt.axes([0.55, 0.005, 0.2, 0.035]) # Posición ajustada
button_finalize = Button(ax_button_finalize, 'Finalizar Simulación', color=DEFAULT_BUTTON_COLOR, hovercolor='0.975')
button_finalize.ax.set_visible(False) # Oculto al inicio

def reset_simulation_state():
    """Resetea la simulación a su estado inicial para una nueva ejecución."""
    global forest, stats_history, im, line_empty, line_tree, line_burning, line_burnt, simulation_started, simulation_paused
    
    simulation_started = False
    simulation_paused = False
    
    forest = np.random.choice([EMPTY, TREE], size=GRID_SIZE, p=[0.1, 0.9])
    im.set_array(forest)
    
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
            im.set_array(forest)
        
        stats_history = [] # Limpiar historial por si acaso
        # Crear la animación AHORA, con blit=False e intervalo ajustado para mayor velocidad real
        # Intervalo reducido de 300ms a 100ms para intentar ~10 cuadros/seg
        ani = animation.FuncAnimation(fig, update_animation, frames=SIMULATION_STEPS, interval=100, blit=False) 
        fig.canvas.draw_idle()
    
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
    
    im.set_array(forest)
    
    x_data = list(range(len(stats_history)))
    line_empty.set_data(x_data, [s['empty'] for s in stats_history])
    line_tree.set_data(x_data, [s['tree'] for s in stats_history])
    line_burning.set_data(x_data, [s['burning'] for s in stats_history])
    line_burnt.set_data(x_data, [s['burnt'] for s in stats_history])
    
    if x_data:
        ax2.set_xlim(0, max(SIMULATION_STEPS * SIMULATION_SPEED_MULTIPLIER, len(x_data)) + 1)

    return [im, line_empty, line_tree, line_burning, line_burnt]

# La animación (ani) NO se crea aquí, sino en start_simulation_callback
# ani = animation.FuncAnimation(fig, update_animation, frames=SIMULATION_STEPS, interval=300, blit=True) # ESTA LÍNEA DEBE ESTAR COMENTADA O ELIMINADA
plt.show() 