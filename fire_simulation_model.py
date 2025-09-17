import numpy as np
import math

# Estados de la celda
EMPTY = 0
GRASS = 1
BURNING = 2
BURNT = 3

# Parámetros de simulación por defecto
DEFAULT_GRID_SIZE = (100, 100)

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

# Factores de influencia para variables climáticas (GRASS -> BURNING)
TEMP_BASELINE_C = 20.0      # Temperatura base para cálculo de efecto
TEMP_SENSITIVITY = 0.005    # Aumento de probabilidad de ignición por °C sobre el baseline
SOIL_MOISTURE_SENSITIVITY = 0.7 # Factor de reducción por humedad del suelo (0 a 1, 1 es reducción total)

# Parámetros configurables
DRYNESS_SPREAD_MULTIPLIER = 0.8  # escala cuánto amplifica la sequedad la propagación
WATER_EFFECT_RADIUS = 3  # celdas
WATER_SOIL_MOISTURE_BONUS = 0.4  # bonus de humedad del suelo por agua cercana (0-1)

# Mapeo explícito del índice del vecino en la lista `neighborhood` a sus coordenadas (dr, dc)
# Este orden debe coincidir con cómo `get_neighborhood` construye la lista.
NEIGHBOR_RELATIVE_COORDS = [
    (-1, -1), (-1, 0), (-1, 1),  # dr = -1
    (0, -1),           (0, 1),   # dr = 0 (centro excluido)
    (1, -1), (1, 0), (1, 1)   # dr = 1
]


class FireSimulationModel:
    """
    Modelo de simulación de incendios forestales basado en autómatas celulares estocásticos.
    """
    
    def __init__(self, grid_size=DEFAULT_GRID_SIZE):
        """
        Inicializa el modelo de simulación.
        
        Args:
            grid_size (tuple): Tamaño de la cuadrícula (filas, columnas)
        """
        self.grid_size = grid_size
        self.forest = np.random.choice([EMPTY, GRASS], size=grid_size, p=[0.1, 0.9])
        self.dryness_grid = np.full(grid_size, 50.0, dtype=float)
        self.water_grid = np.zeros(grid_size, dtype=np.uint8)
        
        # Variables climáticas por defecto
        self.temperature_celsius = 25.0
        self.soil_moisture = 0.5
        self.wind_direction = [-10, 5]  # [componente_y, componente_x]
        self.wind_intensity = 0.9
        self.humidity = 0.3
    
    def get_neighborhood(self, grid, r, c):
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

    def get_transition_matrix(self, neighborhood_states, wind_dir, wind_int, hum, temp_c, soil_moist, current_state, local_dryness_0_to_100=0.0):
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
            local_dryness_0_to_100 (float): Sequedad local [0-100].

        Returns:
            np.array: Vector de probabilidades de transición para el estado actual.
        """
        # Matriz de transición base con valores por defecto
        # Filas: EMPTY, GRASS, BURNING, BURNT
        # Columnas: [EMPTY, GRASS, BURNING, BURNT]
        base_transitions = np.array([
            [1.0 - P_E_G_BASE, P_E_G_BASE, 0.00, 0.00],  # EMPTY
            [0.00, 1.0 - P_G_B_BASE, P_G_B_BASE, 0.00],  # GRASS
            [0.00, 0.00, P_B_B, P_B_X],                 # BURNING
            [P_X_E_BASE, 0.00, 0.00, 1.0 - P_X_E_BASE]  # BURNT
        ])
        
        probs = base_transitions[current_state].copy()
        
        burning_neighbors_count = neighborhood_states.count(BURNING)
        grass_neighbors_count = neighborhood_states.count(GRASS)
        
        # 1. Transiciones desde EMPTY
        if current_state == EMPTY:
            grass_influence = min(F_E_G_NEIGHBOR * grass_neighbors_count, P_E_G_MAX_INFLUENCE)
            probs[GRASS] = P_E_G_BASE + grass_influence
            probs[EMPTY] = 1.0 - probs[GRASS]  # Normalizar
        
        # 2. Transiciones desde GRASS
        elif current_state == GRASS:
            total_wind_factor = 0.0
            if burning_neighbors_count > 0:  # El viento solo importa si hay vecinos quemándose
                wind_y_comp, wind_x_comp = wind_dir[0], wind_dir[1]
                norm_wind = math.sqrt(wind_x_comp**2 + wind_y_comp**2)

                for i, neighbor_state in enumerate(neighborhood_states):
                    if neighbor_state == BURNING:
                        dr_n, dc_n = NEIGHBOR_RELATIVE_COORDS[i]
                        norm_neighbor_vec = math.sqrt(dr_n**2 + dc_n**2)
                        
                        if norm_neighbor_vec > 0 and norm_wind > 0:
                            dot_product = (-dr_n * wind_y_comp) + (-dc_n * wind_x_comp)
                            alignment = dot_product / (norm_neighbor_vec * norm_wind)
                            total_wind_factor += alignment
            
            # Ajustamos por factor de humedad (reduce propagación)
            humidity_reduction_factor = 1.0 - hum
            
            # Efecto de la temperatura
            temp_effect_on_spread = 0.0
            if burning_neighbors_count > 0:
                if temp_c > TEMP_BASELINE_C:
                    temp_effect_on_spread = (temp_c - TEMP_BASELINE_C) * TEMP_SENSITIVITY
            
            # Efecto de la humedad del suelo (reduce propagación)
            soil_moisture_reduction_factor = 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY)
            soil_moisture_reduction_factor = max(0.0, soil_moisture_reduction_factor)
            
            # Probabilidad de quemarse con efecto de sequedad local
            ignition_prob_from_neighbors = F_G_B_NEIGHBOR * burning_neighbors_count
            ignition_prob_from_wind = total_wind_factor * wind_int * F_G_B_WIND
            
            dryness_scale = 1.0 + (max(0.0, min(100.0, float(local_dryness_0_to_100))) / 100.0) * DRYNESS_SPREAD_MULTIPLIER
            effective_spread_prob = (ignition_prob_from_neighbors + ignition_prob_from_wind + temp_effect_on_spread) \
                                    * humidity_reduction_factor * soil_moisture_reduction_factor * dryness_scale
            
            final_burning_prob = P_G_B_BASE + effective_spread_prob
            final_burning_prob = min(max(final_burning_prob, 0.0), P_G_B_MAX)

            probs[BURNING] = final_burning_prob
            probs[GRASS] = 1.0 - final_burning_prob
            if probs[GRASS] < 0: 
                probs[GRASS] = 0
            probs[EMPTY] = 0

        # 3. Transiciones desde BURNING (ya definidas en base_transitions)
        elif current_state == BURNING:
            pass
            
        # 4. Transiciones desde BURNT
        elif current_state == BURNT:
            regen_influence = min(F_X_E_NEIGHBOR * grass_neighbors_count, P_X_E_MAX_INFLUENCE)
            probs[EMPTY] = P_X_E_BASE + regen_influence
            probs[BURNT] = 1.0 - probs[EMPTY]

        # Asegurar que la suma de probabilidades sea 1 y no haya negativas
        probs = np.maximum(probs, 0)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            if current_state == EMPTY: probs[EMPTY] = 1.0
            elif current_state == GRASS: probs[GRASS] = 1.0
            elif current_state == BURNING: probs[BURNING] = 1.0
            elif current_state == BURNT: probs[BURNT] = 1.0
        else:
            probs = probs / probs_sum

        return probs

    def update_step(self):
        """
        Aplica el modelo de incendio basado en cadenas de Markov para actualizar la cuadrícula.

        Returns:
            np.array: La nueva cuadrícula después de una iteración.
        """
        new_grid = self.forest.copy()
        
        for r in range(self.forest.shape[0]):
            for c in range(self.forest.shape[1]):
                # Tratamiento de agua como EMPTY
                if self.water_grid[r, c] > 0:
                    current_cell_state = EMPTY
                else:
                    current_cell_state = self.forest[r, c]
                
                neighborhood = self.get_neighborhood(self.forest, r, c)
                
                local_dryness = 0.0
                try:
                    local_dryness = float(self.dryness_grid[r, c])
                except Exception:
                    local_dryness = 0.0

                # Humedad del suelo local por agua cercana
                r0 = max(0, r - WATER_EFFECT_RADIUS)
                r1 = min(self.grid_size[0], r + WATER_EFFECT_RADIUS + 1)
                c0 = max(0, c - WATER_EFFECT_RADIUS)
                c1 = min(self.grid_size[1], c + WATER_EFFECT_RADIUS + 1)
                near_water = np.any(self.water_grid[r0:r1, c0:c1] > 0)
                local_soil = self.soil_moisture + (WATER_SOIL_MOISTURE_BONUS if near_water else 0.0)
                if local_soil > 1.0:
                    local_soil = 1.0
                    
                transition_probs = self.get_transition_matrix(
                    neighborhood, self.wind_direction, self.wind_intensity, self.humidity,
                    self.temperature_celsius, local_soil,
                    current_cell_state,
                    local_dryness_0_to_100=local_dryness
                )
                
                # Forzar que el agua permanezca EMPTY
                if self.water_grid[r, c] > 0:
                    new_grid[r, c] = EMPTY
                else:
                    new_grid[r, c] = np.random.choice([EMPTY, GRASS, BURNING, BURNT], p=transition_probs)
        
        self.forest = new_grid
        return new_grid

    def compute_statistics(self):
        """
        Calcula las estadísticas de los estados de las celdas en la cuadrícula.

        Returns:
            dict: Un diccionario con el porcentaje de cada tipo de celda.
        """
        total_cells = self.forest.size
        empty_count = np.sum(self.forest == EMPTY)
        grass_count = np.sum(self.forest == GRASS)
        burning_count = np.sum(self.forest == BURNING)
        burnt_count = np.sum(self.forest == BURNT)
        
        return {
            'empty': empty_count / total_cells * 100,
            'grass': grass_count / total_cells * 100,
            'burning': burning_count / total_cells * 100,
            'burnt': burnt_count / total_cells * 100
        }

    def reset_forest(self):
        """Reinicia el bosque a su estado inicial."""
        self.forest = np.random.choice([EMPTY, GRASS], size=self.grid_size, p=[0.1, 0.9])
        self.dryness_grid = np.full(self.grid_size, 50.0, dtype=float)
        self.water_grid = np.zeros(self.grid_size, dtype=np.uint8)

    def set_climate_parameters(self, temperature=None, soil_moisture=None, wind_direction=None, 
                             wind_intensity=None, humidity=None):
        """
        Actualiza los parámetros climáticos del modelo.
        
        Args:
            temperature (float): Temperatura en Celsius
            soil_moisture (float): Humedad del suelo [0-1]
            wind_direction (list): Dirección del viento [componente_y, componente_x]
            wind_intensity (float): Intensidad del viento [0-1]
            humidity (float): Humedad del aire [0-1]
        """
        if temperature is not None:
            self.temperature_celsius = temperature
        if soil_moisture is not None:
            self.soil_moisture = soil_moisture
        if wind_direction is not None:
            self.wind_direction = wind_direction
        if wind_intensity is not None:
            self.wind_intensity = wind_intensity
        if humidity is not None:
            self.humidity = humidity

    def apply_brush(self, r_center, c_center, brush_size, brush_type, value=None):
        """
        Aplica un pincel en la cuadrícula.
        
        Args:
            r_center (int): Fila central del pincel
            c_center (int): Columna central del pincel
            brush_size (int): Tamaño del pincel en celdas
            brush_type (str): Tipo de pincel ('fire', 'grass', 'empty', 'water', 'dryness')
            value (float): Valor para el pincel de sequedad [0-100]
        """
        half = brush_size // 2
        r_start = max(0, r_center - half)
        r_end = min(self.grid_size[0], r_start + brush_size)
        c_start = max(0, c_center - half)
        c_end = min(self.grid_size[1], c_start + brush_size)
        
        if brush_type == 'fire':
            self.forest[r_start:r_end, c_start:c_end] = BURNING
        elif brush_type == 'grass':
            self.forest[r_start:r_end, c_start:c_end] = GRASS
        elif brush_type == 'empty':
            self.forest[r_start:r_end, c_start:c_end] = EMPTY
        elif brush_type == 'water':
            self.water_grid[r_start:r_end, c_start:c_end] = 1
        elif brush_type == 'dryness' and value is not None:
            self.dryness_grid[r_start:r_end, c_start:c_end] = float(value)
