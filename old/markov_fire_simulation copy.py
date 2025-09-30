import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import math

# Estados de la celda
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3

# Parámetros de simulación
grid_size = (100, 100)  # Tamaño de la cuadrícula
prob_spread_base = 0.6  # Probabilidad base de propagación
wind_direction = (-10, 5)  # Dirección del viento (y, x)
wind_intensity = 0.9  # Intensidad del viento (0 a 1)
humidity = 0.3  # Factor de humedad (reduce propagación)

# Función para obtener los estados de las celdas vecinas
def get_neighborhood(grid, x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # Excluimos la celda central
                continue
            nx, ny = x + dx, y + dy
            # Manejo de bordes
            if nx < 0 or ny < 0 or nx >= grid.shape[0] or ny >= grid.shape[1]:
                neighbors.append(EMPTY)  # Consideramos borde como vacío
            else:
                neighbors.append(grid[nx, ny])
    return neighbors

# Función para calcular la matriz de transición específica para cada configuración
def get_transition_matrix(neighborhood, wind_direction, humidity, current_state):
    # Base transition matrix - valores por defecto
    transition_matrix = np.array([
        [0.98, 0.02, 0.00, 0.00],  # EMPTY -> [EMPTY, TREE, BURNING, BURNT]
        [0.00, 0.95, 0.05, 0.00],  # TREE -> [EMPTY, TREE, BURNING, BURNT]
        [0.00, 0.00, 0.10, 0.90],  # BURNING -> [EMPTY, TREE, BURNING, BURNT]
        [0.05, 0.00, 0.00, 0.95]   # BURNT -> [EMPTY, TREE, BURNING, BURNT]
    ])
    
    # Contamos cuántos vecinos están en cada estado
    burning_neighbors = neighborhood.count(BURNING)
    tree_neighbors = neighborhood.count(TREE)
    
    # Factores que modifican las probabilidades de transición
    
    # 1. Estado EMPTY: más probabilidad de convertirse en TREE si hay árboles cerca
    if current_state == EMPTY:
        tree_influence = min(0.05 * tree_neighbors, 0.3)  # Máximo 30% de probabilidad
        transition_matrix[EMPTY][TREE] = 0.02 + tree_influence
        transition_matrix[EMPTY][EMPTY] = 1.0 - transition_matrix[EMPTY][TREE]  # Normalizar
    
    # 2. Estado TREE: más probabilidad de incendiarse con vecinos quemándose
    if current_state == TREE:
        # Calculamos influencia direccional del viento
        wind_influence = 0
        for i, neighbor in enumerate(neighborhood):
            if neighbor == BURNING:
                # Calculamos posición relativa (-1, -1), (-1, 0), etc.
                dx = (i % 3) - 1
                dy = (i // 3) - 1
                
                # Producto escalar para determinar alineación con el viento
                dot_product = dx * wind_direction[1] + dy * wind_direction[0]
                magnitude = math.sqrt(dx**2 + dy**2) * math.sqrt(wind_direction[0]**2 + wind_direction[1]**2)
                
                # Evitar división por cero
                if magnitude > 0:
                    alignment = dot_product / magnitude
                    # Mayor contribución si el viento sopla desde el vecino hacia la celda actual
                    wind_influence += alignment * wind_intensity
        
        # Ajustamos por factor de humedad (reduce propagación)
        humidity_factor = 1 - humidity
        
        # Probabilidad base + influencia de vecinos quemándose + efecto del viento
        burning_prob = min(0.05 + (0.15 * burning_neighbors * humidity_factor) + (wind_influence * 0.1), 0.95)
        
        transition_matrix[TREE][BURNING] = burning_prob
        transition_matrix[TREE][TREE] = 1.0 - burning_prob  # Normalizar
    
    # 3. Estado BURNING: siempre se quema completamente
    if current_state == BURNING:
        # No cambiamos, mantenemos alta probabilidad de pasar a BURNT
        pass
        
    # 4. Estado BURNT: regeneración gradual
    if current_state == BURNT:
        # Más probabilidad de regenerarse si hay árboles alrededor (semillas)
        regen_influence = min(0.02 * tree_neighbors, 0.2)
        transition_matrix[BURNT][EMPTY] = 0.05 + regen_influence
        transition_matrix[BURNT][BURNT] = 1.0 - transition_matrix[BURNT][EMPTY]  # Normalizar
    
    return transition_matrix[current_state]

# Modelo de incendio puramente basado en Markov
def pure_markov_fire_model(grid):
    new_grid = grid.copy()
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            # Extraer configuración de vecinos
            neighborhood = get_neighborhood(grid, x, y)
            
            # Obtener probabilidades de transición específicas para esta configuración
            current_state = grid[x, y]
            transition_probs = get_transition_matrix(
                neighborhood, wind_direction, humidity, current_state
            )
            
            # Aplicar transición de Markov
            new_grid[x, y] = np.random.choice([EMPTY, TREE, BURNING, BURNT], p=transition_probs)
    
    return new_grid

# Inicializa la cuadrícula con árboles
forest = np.random.choice([EMPTY, TREE], size=grid_size, p=[0.1, 0.9])
# Fuego inicial en el centro
forest[grid_size[0] // 2, grid_size[1] // 2] = BURNING

# Para mostrar estadísticas de la simulación
def compute_statistics(grid):
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

# Animación de la propagación del incendio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
im = ax1.imshow(forest, cmap=cmap, vmin=0, vmax=3)
ax1.set_title("Simulación de Incendio Forestal (Modelo Markov Puro)")

# Para el gráfico de estadísticas
line_empty, = ax2.plot([], [], 'w-', label='Vacío')
line_tree, = ax2.plot([], [], 'g-', label='Árbol')
line_burning, = ax2.plot([], [], 'r-', label='Quemando')
line_burnt, = ax2.plot([], [], 'k-', label='Quemado')
ax2.set_xlim(0, 50)  # Ajustar según número de frames
ax2.set_ylim(0, 100)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Porcentaje')
ax2.legend()
ax2.set_title("Evolución de Estados")
ax2.grid(True)

def update(frame):
    global forest, stats_history
    forest = pure_markov_fire_model(forest)
    im.set_array(forest)
    
    # Actualizar estadísticas
    stats = compute_statistics(forest)
    stats_history.append(stats)
    
    # Actualizar gráfico de estadísticas
    x = list(range(len(stats_history)))
    line_empty.set_data(x, [s['empty'] for s in stats_history])
    line_tree.set_data(x, [s['tree'] for s in stats_history])
    line_burning.set_data(x, [s['burning'] for s in stats_history])
    line_burnt.set_data(x, [s['burnt'] for s in stats_history])
    
    return [im, line_empty, line_tree, line_burning, line_burnt]

ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
plt.tight_layout()
plt.show() 