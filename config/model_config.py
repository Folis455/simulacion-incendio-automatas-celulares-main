import numpy as np

# Estados de la celda
EMPTY = 0
GRASS = 1
BURNING = 2
BURNT = 3

# Parámetros de simulación por defecto
DEFAULT_GRID_SIZE = (100, 100)
DEFAULT_EMPTY_PROB = 0.1
DEFAULT_GRASS_PROB = 0.9
DEFAULT_GRASS_DRYNESS = 50.0  # Sequedad del pasto
DEFAULT_TEMPERATURE = 25.0
DEFAULT_SOIL_MOISTURE = 0.5  # Humedad del suelo
DEFAULT_WIND_DIRECTION = [-10, 5]  # [componente_y, componente_x]
DEFAULT_WIND_INTENSITY = 0.9
DEFAULT_HUMIDITY = 0.3  # Humedad ambiente
DEFAULT_GRASS_DENSITY = 0.375  # Humedad ambiente

# Probabilidades base y factores de influencia (para get_transition_matrix)
# EMPTY state transitions
P_E_G_BASE = 0.00000  # Probabilidad base EMPTY -> GRASS (REDUCIDO DRÁSTICAMENTE de 0.02)
F_E_G_NEIGHBOR = 0.00  # Factor de influencia de vecinos GRASS para EMPTY -> GRASS
P_E_G_MAX_INFLUENCE = 0.00  # Máxima probabilidad adicional por influencia para EMPTY -> GRASS (REDUCIDO de 0.3)

# GRASS state transitions
P_G_B_BASE = 0.00000  # Probabilidad base GRASS -> BURNING (REDUCIDO DRÁSTICAMENTE de 0.05)
F_G_B_NEIGHBOR = 0.15  # Factor de propagación por vecinos BURNING para GRASS -> BURNING
F_G_B_WIND = 0.35  # Factor de escala para la influencia del viento en GRASS -> BURNING (AUMENTADO de 0.1)
P_G_B_MAX = 0.95  # Máxima probabilidad para GRASS -> BURNING

# BURNING state transitions
P_B_B = 0.80  # Probabilidad BURNING -> BURNING (permanecer quemándose) (AUMENTADO de 0.10)
P_B_X = 0.20  # Probabilidad BURNING -> BURNT (X para quemado) (REDUCIDO de 0.90)

# BURNT state transitions
P_X_E_BASE = 0.00000  # Probabilidad base BURNT -> EMPTY (regeneración) (REDUCIDO DRÁSTICAMENTE de 0.05)
F_X_E_NEIGHBOR = 0.00  # Factor de influencia de vecinos GRASS para BURNT -> EMPTY (semillas)
P_X_E_MAX_INFLUENCE = 0.0  # Máxima probabilidad adicional por influencia para BURNT -> EMPTY

# Matriz de transición base con valores por defecto
# Filas: EMPTY, GRASS, BURNING, BURNT
# Columnas: [EMPTY, GRASS, BURNING, BURNT]
BASE_TRANSITIONS = np.array([
    [1.0 - P_E_G_BASE, P_E_G_BASE, 0.00, 0.00],  # EMPTY
    [0.00, 1.0 - P_G_B_BASE, P_G_B_BASE, 0.00],  # GRASS
    [0.00, 0.00, P_B_B, P_B_X],  # BURNING
    [P_X_E_BASE, 0.00, 0.00, 1.0 - P_X_E_BASE]  # BURNT
])

# Factores de influencia para variables climáticas (GRASS -> BURNING)
TEMPERATURE_BASELINE = 20.0  # Temperatura base para cálculo de efecto
TEMPERATURE_SENSITIVITY = 0.005  # Aumento de probabilidad de ignición por °C sobre el baseline
SOIL_MOISTURE_SENSITIVITY = 0.7  # Factor de reducción por humedad del suelo (0 a 1, 1 es reducción total)

# Parámetros configurables
DRYNESS_SPREAD_MULTIPLIER = 0.8  # escala cuánto amplifica la sequedad la propagación
WATER_EFFECT_RADIUS = 3  # celdas
WATER_SOIL_MOISTURE_BONUS = 0.4  # bonus de humedad del suelo por agua cercana (0-1)

# Mapeo explícito del índice del vecino en la lista `neighborhood` a sus coordenadas (dr, dc)
# Este orden debe coincidir con cómo `get_neighborhood` construye la lista.
NEIGHBOR_RELATIVE_COORDS = [
    (-1, -1), (-1, 0), (-1, 1),  # dr = -1
    (0, -1), (0, 1),  # dr = 0 (centro excluido)
    (1, -1), (1, 0), (1, 1)  # dr = 1
]
