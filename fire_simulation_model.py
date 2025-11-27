import numpy as np
import math
from config.model_config import (
    EMPTY, GRASS, BURNING, BURNT,
    DEFAULT_GRID_SIZE, DEFAULT_EMPTY_PROB, DEFAULT_GRASS_PROB, DEFAULT_GRASS_DRYNESS, DEFAULT_TEMPERATURE, DEFAULT_SOIL_MOISTURE,
    DEFAULT_WIND_DIRECTION, DEFAULT_WIND_INTENSITY, DEFAULT_HUMIDITY,
    P_G_B_BASE, F_G_B_NEIGHBOR, F_G_B_WIND, P_G_B_MAX,
    TEMPERATURE_BASELINE, TEMPERATURE_SENSITIVITY, SOIL_MOISTURE_SENSITIVITY,
    DRYNESS_SPREAD_MULTIPLIER, WATER_EFFECT_RADIUS, WATER_SOIL_MOISTURE_BONUS,
    NEIGHBOR_RELATIVE_COORDS, BASE_TRANSITIONS, DEFAULT_GRASS_DENSITY
)


class FireSimulationModel:
    """
    Modelo de simulación de incendios en pastizales basado en autómatas celulares estocásticos.
    """

    def __init__(self) -> None:
        """
        Inicializa el modelo de simulación.
        """
        self.grid_size = DEFAULT_GRID_SIZE
        self.land = np.random.choice([EMPTY, GRASS], size=self.grid_size, p=[DEFAULT_EMPTY_PROB, DEFAULT_GRASS_PROB])
        self.dryness_grid = np.full(self.grid_size, DEFAULT_GRASS_DRYNESS, dtype=float)
        self.water_grid = np.zeros(self.grid_size, dtype=np.uint8)

        # Variables climáticas por defecto
        self.temperature = DEFAULT_TEMPERATURE
        self.soil_moisture = DEFAULT_SOIL_MOISTURE
        self.wind_direction = DEFAULT_WIND_DIRECTION
        self.wind_intensity = DEFAULT_WIND_INTENSITY
        self.humidity = DEFAULT_HUMIDITY
        self.grass_density = float(min(max(DEFAULT_GRASS_DENSITY, 0.0), 1.0))
        self.near_water_grid = np.zeros(self.grid_size, dtype=bool)
        self.calculate_water_effect()

    def calculate_water_effect(self) -> None:
        rows, cols = self.grid_size

        print("precalculando...")

        for r in range(rows):
            for c in range(cols):
                r0 = max(0, r - WATER_EFFECT_RADIUS)
                r1 = min(rows, r + WATER_EFFECT_RADIUS + 1)
                c0 = max(0, c - WATER_EFFECT_RADIUS)
                c1 = min(cols, c + WATER_EFFECT_RADIUS + 1)

                if np.any(self.water_grid[r0:r1, c0:c1] > 0):
                    self.near_water_grid[r, c] = True
                else:
                    self.near_water_grid[r, c] = False

    def get_neighborhood(self, r: int, c: int) -> list[int]:
        """
        Obtiene los estados de las 8 celdas vecinas (vecindad de Moore).
        Si un vecino está fuera de la cuadrícula, se considera EMPTY.

        Args:
            r (int): Fila de la celda central.
            c (int): Columna de la celda central.

        Returns: Una lista con los estados de las 8 celdas vecinas.
                 El orden es: (-1,-1),(-1,0),(-1,1), (0,-1),(0,1), (1,-1),(1,0),(1,1) relativo a la celda (r,c),
                 donde dr es el primer elemento del par.
        """
        neighbors = []
        for dr in [-1, 0, 1]:  # Delta fila
            for dc in [-1, 0, 1]:  # Delta columna
                if dr == 0 and dc == 0:  # Excluimos la celda central
                    continue
                nr, nc = r + dr, c + dc
                # Manejo de bordes
                if nr < 0 or nc < 0 or nr >= self.land.shape[0] or nc >= self.land.shape[1]:
                    neighbors.append(EMPTY)  # Consideramos borde como vacío
                else:
                    neighbors.append(self.land[nr, nc])
        return neighbors

    def get_transition_matrix(self, neighborhood_states: list[int], wind_direction: list[int], wind_intensity: float,
                              humidity: float, temperature: float, soil_moist: float, current_state: int,
                              local_dryness: float = 0.0) -> np.ndarray:
        """
        Calcula las probabilidades de transición para una celda, dado su estado actual
        y el de sus vecinos, además de las condiciones climáticas.

        Args:
            neighborhood_states (list): Estados de las celdas vecinas.
            wind_direction (list): Dirección del viento [componente_y, componente_x].
            wind_intensity (float): Intensidad del viento.
            humidity (float): Humedad del aire.
            temperature (float): Temperatura en Celsius.
            soil_moist (float): Humedad del suelo.
            current_state (int): Estado actual de la celda.
            local_dryness (float): Sequedad local [0-100].

        Returns: np.array: Vector de probabilidades de transición para el estado actual.
        """
        probs = BASE_TRANSITIONS[current_state].copy()
        burning_neighbors_count = neighborhood_states.count(BURNING)

        # 1. Transiciones desde GRASS
        if current_state == GRASS:
            total_wind_factor = 0.0
            if burning_neighbors_count > 0:  # El viento solo importa si hay vecinos quemándose
                wind_y_comp, wind_x_comp = wind_direction[0], wind_direction[1]
                norm_wind = math.hypot(wind_y_comp, wind_x_comp)

                for i, neighbor_state in enumerate(neighborhood_states):
                    if neighbor_state == BURNING:
                        dr_n, dc_n = NEIGHBOR_RELATIVE_COORDS[i]
                        norm_neighbor_vec = math.hypot(dr_n, dc_n)

                        if norm_neighbor_vec > 0 and norm_wind > 0:
                            dot_product = (-dr_n * wind_y_comp) + (-dc_n * wind_x_comp)
                            alignment = dot_product / (norm_neighbor_vec * norm_wind)
                            total_wind_factor += alignment

            # Ajustamos por factor de humedad (reduce propagación)
            humidity_reduction_factor = 1.0 - humidity
            soil_moisture_reduction_factor = 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY)
            soil_moisture_reduction_factor = max(0.0, soil_moisture_reduction_factor)

            # Efecto de la temperatura
            temp_effect_on_spread = 0.0
            if burning_neighbors_count > 0:
                if temperature > TEMPERATURE_BASELINE:
                    temp_effect_on_spread = (temperature - TEMPERATURE_BASELINE) * TEMPERATURE_SENSITIVITY

            # Probabilidad de quemarse con efecto de sequedad local
            ignition_prob_from_neighbors = F_G_B_NEIGHBOR * burning_neighbors_count
            ignition_prob_from_wind = total_wind_factor * (wind_intensity / 2) * F_G_B_WIND
            dryness_scale = 1.0 + (max(0.0, min(100.0, local_dryness)) / 100.0) * DRYNESS_SPREAD_MULTIPLIER

            effective_spread_prob = (ignition_prob_from_neighbors + ignition_prob_from_wind + temp_effect_on_spread) \
                                    * humidity_reduction_factor * soil_moisture_reduction_factor * dryness_scale

            final_burning_prob = min(max(P_G_B_BASE + effective_spread_prob, 0.0), P_G_B_MAX)

            probs[BURNING] = final_burning_prob
            probs[GRASS] = max(0.0, 1.0 - final_burning_prob)
            probs[EMPTY] = 0

        # 2. Transiciones desde BURNING (ajuste dinámico por densidad)
        elif current_state == BURNING:
            # Duración esperada E en [2,10]; mayor densidad => mayor duración
            expected_burn_steps = 2.0 + self.grass_density * (10.0 - 2.0)
            p_burn_to_burnt = 1.0 / max(2.0, expected_burn_steps)
            p_burning_to_burning = max(0.0, 1.0 - p_burn_to_burnt)
            probs[BURNING] = p_burning_to_burning
            probs[BURNT] = p_burn_to_burnt

        # Asegurar que la suma de probabilidades sea 1 y no haya negativas
        probs = np.maximum(probs, 0)
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            if current_state == EMPTY:
                probs[EMPTY] = 1.0
            elif current_state == GRASS:
                probs[GRASS] = 1.0
            elif current_state == BURNING:
                probs[BURNING] = 1.0
            elif current_state == BURNT:
                probs[BURNT] = 1.0
        else:
            probs = probs / probs_sum

        return probs

    def update_step(self) -> np.ndarray:
        """
        Aplica el modelo de incendio basado en cadenas de Markov para actualizar la cuadrícula.

        Returns:
            np.array: La nueva cuadrícula después de una iteración.
        """
        new_grid = self.land.copy()
        rows, cols = self.land.shape

        for r in range(rows):  # Bucle lento
            for c in range(cols):  # Bucle lento
                # Tratamiento de agua como EMPTY
                current_cell_state = EMPTY if self.water_grid[r, c] > 0 else self.land[r, c]

                neighborhood_states = self.get_neighborhood(r, c)

                try:
                    local_dryness = float(self.dryness_grid[r, c])
                except Exception:
                    local_dryness = 0.0

                # near_water = np.any(self.water_grid[r0:r1, c0:c1] > 0) # MUY lento
                near_water = self.near_water_grid[r, c]  # Usando los datos precalculados
                local_soil_moist = min(self.soil_moisture + (WATER_SOIL_MOISTURE_BONUS if near_water else 0.0), 1.0)

                transition_probs = self.get_transition_matrix(
                    neighborhood_states=neighborhood_states,
                    wind_direction=self.wind_direction,
                    wind_intensity=self.wind_intensity,
                    humidity=self.humidity,
                    temperature=self.temperature,
                    soil_moist=local_soil_moist,
                    current_state=current_cell_state,
                    local_dryness=local_dryness
                )

                # Forzar que el agua permanezca EMPTY
                new_grid[r, c] = EMPTY if self.water_grid[r, c] > 0 else np.random.choice([EMPTY, GRASS, BURNING, BURNT],
                                                                                          p=transition_probs)

        self.land = new_grid
        return new_grid

    def compute_statistics(self) -> dict[str, float]:
        """
        Calcula las estadísticas de los estados de las celdas en la cuadrícula.

        Returns:
            dict: Un diccionario con el porcentaje de cada tipo de celda.
        """
        total_cells = self.land.size
        empty_count = np.sum(self.land == EMPTY)
        grass_count = np.sum(self.land == GRASS)
        burning_count = np.sum(self.land == BURNING)
        burnt_count = np.sum(self.land == BURNT)

        return {
            'empty': empty_count / total_cells * 100,
            'grass': grass_count / total_cells * 100,
            'burning': burning_count / total_cells * 100,
            'burnt': burnt_count / total_cells * 100
        }

    def get_burned_mask(self) -> np.ndarray:
        """
        Devuelve una máscara binaria de celdas quemadas o en combustión.

        Returns:
            np.ndarray: array booleano con True donde land es BURNING o BURNT.
        """
        return (self.land == BURNING) | (self.land == BURNT)

    def apply_brush(self, r_center: int, c_center: int, brush_size: int, brush_type: str, value: float | None = None) -> None:
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
            self.land[r_start:r_end, c_start:c_end] = BURNING
        elif brush_type == 'grass':
            self.land[r_start:r_end, c_start:c_end] = GRASS
        elif brush_type == 'empty':
            self.land[r_start:r_end, c_start:c_end] = EMPTY
            self.water_grid[r_start:r_end, c_start:c_end] = 0
        elif brush_type == 'water':
            self.water_grid[r_start:r_end, c_start:c_end] = 1
        elif brush_type == 'dryness' and value is not None:
            self.dryness_grid[r_start:r_end, c_start:c_end] = float(value)

    def import_state(self, data: dict) -> None:
        self.grid_size = tuple(data["grid_size"])
        self.land = np.array(data["land"], dtype=np.uint8)
        self.dryness_grid = np.array(data["dryness_grid"], dtype=float)
        self.water_grid = np.array(data["water_grid"], dtype=np.uint8)
        self.temperature = float(data["temperature"])
        self.soil_moisture = float(data["soil_moisture"])
        self.wind_direction = list(data["wind_direction"])
        self.wind_intensity = float(data["wind_intensity"])
        self.humidity = float(data["humidity"])
        self.grass_density = float(data["grass_density"])
        self.calculate_water_effect()
