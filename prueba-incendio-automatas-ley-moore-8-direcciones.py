import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import math

# Parámetros iniciales
grid_size = (200, 200)  # Tamaño de la cuadrícula
prob_spread = 0.7  # Probabilidad base de propagación
wind_direction = (-10, 5)  #Dirección del viento (y, x) (el y negativo es hacia arriba, matplotlib lo toma al reves que un eje cartesiano)
                           #la direccion del viento va cambiando con respecto al tiempo, no dejarlo fijo
wind_intensity = 0.9  # Intensidad del viento (0 a 1)
                      # la potencia del viento va cambiando con respecto al tiempo, no dejarlo fijo
humidity = 0.3  # Factor de humedad (reduce propagación)

# Estados de la celda
EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3

# Definir colores personalizados para cada estado
colors = ["white", "green", "red", "black"]  # Vacío, Árbol, Fuego, Quemado
cmap = ListedColormap(colors)

# Matriz de transición de Markov para estados de celdas
# Las filas representan el estado actual, las columnas el estado siguiente
# [EMPTY->EMPTY, EMPTY->TREE, EMPTY->BURNING, EMPTY->BURNT]
# [TREE->EMPTY, TREE->TREE, TREE->BURNING, TREE->BURNT]
# [BURNING->EMPTY, BURNING->TREE, BURNING->BURNING, BURNING->BURNT]
# [BURNT->EMPTY, BURNT->TREE, BURNT->BURNING, BURNT->BURNT]
markov_matrix = np.array([
    [0.98, 0.02, 0.00, 0.00],  # EMPTY casi siempre se queda igual, pequeña prob de crecer un árbol
    [0.00, 0.95, 0.05, 0.00],  # TREE puede permanecer o empezar a quemarse
    [0.00, 0.00, 0.10, 0.90],  # BURNING casi siempre pasa a BURNT
    [0.05, 0.00, 0.00, 0.95]   # BURNT puede regenerarse a EMPTY con baja probabilidad
])

# Función para aplicar transiciones de Markov en una celda específica
def apply_markov_transition(state, influence_factor=0.2):
    # La transición base viene de la matriz de Markov
    transition_probs = markov_matrix[state]
    
    # El factor de influencia determina cuánto aplicamos la matriz de Markov vs. las reglas del autómata
    if np.random.rand() < influence_factor:
        return np.random.choice([EMPTY, TREE, BURNING, BURNT], p=transition_probs)
    else:
        return state  # Devolvemos el mismo estado para que las reglas del autómata lo manejen

# Inicializa la cuadrícula con árboles
forest = np.random.choice([EMPTY, TREE], size=grid_size, p=[0.1, 0.9])
forest[grid_size[0] // 2, grid_size[1] // 2] = BURNING  # Fuego inicial en el centro

def spread_fire_moore(grid):
    new_grid = grid.copy()
    wind_x, wind_y = wind_direction  # Dirección del viento
    wind_factor = 0.7  # Factor de influencia del viento (reduce impacto directo)

    # Primero aplicamos las transiciones de Markov globalmente
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            new_grid[x, y] = apply_markov_transition(grid[x, y])
    
    # Luego aplicamos las reglas del autómata celular para la propagación del fuego
    for x in range(1, grid.shape[0] - 1):
        for y in range(1, grid.shape[1] - 1):
            if grid[x, y] == BURNING:
                new_grid[x, y] = BURNT  # Se quema completamente
                
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), 
                               (0, -1), (0, 1), 
                               (1, -1), (1, 0), (1, 1)]:  # Vecindad de Moore (8 celdas)
                    if grid[x + dx, y + dy] == TREE:
                        # Probabilidad base ajustada por humedad
                        spread_chance = prob_spread * (1 - humidity)
                        
                        # Influencia del viento según ángulo (más moderada)
                        dot_product = dx * wind_x + dy * wind_y
                        wind_effect = (dot_product / (math.sqrt(dx**2 + dy**2 + 1))) * wind_intensity
                        wind_effect = max(min(wind_effect, 0.2), -0.2)  # Limitar efecto del viento
                        
                        spread_chance += wind_effect * wind_factor  # Moderar el impacto
                        
                        if np.random.rand() < spread_chance:
                            new_grid[x + dx, y + dy] = BURNING
    
    return new_grid

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

# Animación de la propagación del incendio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
im = ax1.imshow(forest, cmap=cmap, vmin=0, vmax=3)  # Aplica el nuevo mapa de colores
ax1.set_title("Simulación de Incendio Forestal con Cadenas de Markov")

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

def update_moore(frame):
    global forest, stats_history
    forest = spread_fire_moore(forest)
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

ani = animation.FuncAnimation(fig, update_moore, frames=50, interval=200, blit=True)
plt.tight_layout()
plt.show()
