import numpy as np
import math
from numba import jit
from config.model_config import (
    EMPTY, GRASS, BURNING, BURNT,
    DEFAULT_GRID_SIZE, DEFAULT_EMPTY_PROB, DEFAULT_GRASS_PROB, DEFAULT_GRASS_DRYNESS, DEFAULT_TEMPERATURE, DEFAULT_SOIL_MOISTURE,
    DEFAULT_WIND_DIRECTION, DEFAULT_WIND_INTENSITY, DEFAULT_HUMIDITY,
    P_G_B_BASE, F_G_B_NEIGHBOR, F_G_B_WIND, P_G_B_MAX,
    TEMPERATURE_BASELINE, TEMPERATURE_SENSITIVITY, SOIL_MOISTURE_SENSITIVITY,
    DRYNESS_SPREAD_MULTIPLIER, WATER_EFFECT_RADIUS, WATER_SOIL_MOISTURE_BONUS,
    NEIGHBOR_RELATIVE_COORDS, BASE_TRANSITIONS, DEFAULT_GRASS_DENSITY
)

# Definir variables localmente para reducir el lookup y que numba las pueda acceder más rápidamente
NEIGHBOR_OFFSETS = np.array(NEIGHBOR_RELATIVE_COORDS, dtype=np.int8)
STATES = np.array([EMPTY, GRASS, BURNING, BURNT], dtype=np.int64)


@jit(nopython=True, fastmath=True)
def calculate_water_effect(water_grid, rows, cols):
    near_water_grid = np.zeros((rows, cols), dtype=np.bool_)
    radius = WATER_EFFECT_RADIUS

    for r in range(rows):
        for c in range(cols):
            found = False
            r0 = max(0, r - radius)
            r1 = min(rows, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(cols, c + radius + 1)

            for ir in range(r0, r1):
                for ic in range(c0, c1):
                    if water_grid[ir, ic] > 0:
                        found = True
                        break
                if found: break

            near_water_grid[r, c] = found
    return near_water_grid


@jit(nopython=True)
def get_neighborhood(r, c, land_grid):
    neighbors = np.empty(8, dtype=np.int64)
    rows, cols = land_grid.shape

    for i in range(8):
        dr = NEIGHBOR_OFFSETS[i, 0]
        dc = NEIGHBOR_OFFSETS[i, 1]
        nr, nc = r + dr, c + dc

        if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
            neighbors[i] = EMPTY
        else:
            neighbors[i] = land_grid[nr, nc]

    return neighbors


@jit(nopython=True, fastmath=True)
def _jit_get_transition_probs(neighborhood_states, wind_direction, wind_intensity,
                              humidity, temperature, soil_moist, current_state,
                              local_dryness, grass_density):
    # Copiamos las transiciones base
    probs = np.zeros(4, dtype=np.float64)
    # Copia manual de BASE_TRANSITIONS para evitar allocs, asumiendo estructura conocida
    if current_state == EMPTY:
        probs[0] = 1.0
    elif current_state == GRASS:
        probs[1] = BASE_TRANSITIONS[1, 1]  # GRASS -> GRASS
        probs[2] = BASE_TRANSITIONS[1, 2]  # GRASS -> BURNING
    elif current_state == BURNING:
        probs[2] = BASE_TRANSITIONS[2, 2]  # BURNING -> BURNING
        probs[3] = BASE_TRANSITIONS[2, 3]  # BURNING -> BURNT
    elif current_state == BURNT:
        probs[3] = 1.0

    # Contar vecinos quemándose
    burning_neighbors_count = 0
    for s in neighborhood_states:
        if s == BURNING:
            burning_neighbors_count += 1

    # 1. Transiciones desde GRASS
    if current_state == GRASS:
        total_wind_factor = 0.0
        if burning_neighbors_count > 0:
            wind_y, wind_x = wind_direction[0], wind_direction[1]
            norm_wind = math.hypot(wind_y, wind_x)

            for i in range(8):
                if neighborhood_states[i] == BURNING:
                    # Usamos el array de offsets predefinido
                    dr_n = NEIGHBOR_OFFSETS[i, 0]
                    dc_n = NEIGHBOR_OFFSETS[i, 1]
                    norm_neighbor = math.hypot(dr_n, dc_n)

                    if norm_neighbor > 0 and norm_wind > 0:
                        dot = (-dr_n * wind_y) + (-dc_n * wind_x)
                        alignment = dot / (norm_neighbor * norm_wind)
                        total_wind_factor += alignment

        humidity_reduction = 1.0 - humidity
        soil_reduction = max(0.0, 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY))

        temp_effect = 0.0
        if burning_neighbors_count > 0 and temperature > TEMPERATURE_BASELINE:
            temp_effect = (temperature - TEMPERATURE_BASELINE) * TEMPERATURE_SENSITIVITY

        prob_neighbors = F_G_B_NEIGHBOR * burning_neighbors_count
        prob_wind = total_wind_factor * wind_intensity * F_G_B_WIND
        dryness_scale = 1.0 + (max(0.0, min(100.0, local_dryness)) / 100.0) * DRYNESS_SPREAD_MULTIPLIER

        effective_prob = (prob_neighbors + prob_wind + temp_effect) * \
                         humidity_reduction * soil_reduction * dryness_scale

        final_prob = min(max(P_G_B_BASE + effective_prob, 0.0), P_G_B_MAX)

        probs[BURNING] = final_prob
        probs[GRASS] = max(0.0, 1.0 - final_prob)
        probs[EMPTY] = 0.0

    # 2. Transiciones desde BURNING
    elif current_state == BURNING:
        expected_steps = 2.0 + grass_density * (10.0 - 2.0)
        p_to_burnt = 1.0 / max(2.0, expected_steps)
        p_stay_burning = max(0.0, 1.0 - p_to_burnt)

        probs[BURNING] = p_stay_burning
        probs[BURNT] = p_to_burnt

    # Normalización segura
    probs_sum = 0.0
    for p in probs:
        probs_sum += p

    if probs_sum > 0:
        for i in range(4):
            probs[i] /= probs_sum
    else:
        probs[current_state] = 1.0

    return probs


@jit(nopython=True, fastmath=True)
def _jit_update_loop(rows, cols, land_grid, water_grid, dryness_grid, near_water_grid,
                     wind_dir, wind_int, humidity, temp, soil_moist, grass_density):
    new_grid = land_grid.copy()

    for r in range(rows):
        for c in range(cols):
            # Si hay agua en la celda, forzamos EMPTY
            if water_grid[r, c] > 0:
                new_grid[r, c] = EMPTY
                continue

            current_cell_state = land_grid[r, c]

            # Obtenemos vecindad (versión array)
            neighborhood = get_neighborhood(r, c, land_grid)

            # Sequedad local (acceso directo sin try/except, es seguro en Numba si los arrays coinciden)
            local_dryness = dryness_grid[r, c]

            # Humedad suelo
            is_near_water = near_water_grid[r, c]
            bonus = WATER_SOIL_MOISTURE_BONUS if is_near_water else 0.0
            local_soil_moist = soil_moist + bonus
            if local_soil_moist > 1.0: local_soil_moist = 1.0

            # Obtener probabilidades
            probs = _jit_get_transition_probs(
                neighborhood, wind_dir, wind_int, humidity, temp,
                local_soil_moist, current_cell_state, local_dryness, grass_density
            )

            # Selección aleatoria manual (más rápido que np.random.choice en loop)
            rand_val = np.random.random()
            cumulative = 0.0
            choice = EMPTY
            for i in range(4):
                cumulative += probs[i]
                if rand_val < cumulative:
                    choice = STATES[i]
                    break

            new_grid[r, c] = choice

    return new_grid


# --- CLASE ORIGINAL (WRAPPER) ---

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
        self.wind_direction = list(DEFAULT_WIND_DIRECTION)  # Aseguramos lista para compatibilidad
        self.wind_intensity = DEFAULT_WIND_INTENSITY
        self.humidity = DEFAULT_HUMIDITY
        self.grass_density = float(min(max(DEFAULT_GRASS_DENSITY, 0.0), 1.0))
        self.near_water_grid = np.zeros(self.grid_size, dtype=bool)

        self.calculate_water_effect()

        # Warmup de Numba (ejecuta una vez con datos dummy para compilar)
        self._numba_warmup()

    def _numba_warmup(self):
        # Ejecución silenciosa para compilar las funciones JIT al iniciar
        calculate_water_effect(np.zeros((2, 2), dtype=np.uint8), 2, 2)

    def calculate_water_effect(self) -> None:
        rows, cols = self.grid_size
        # Delegamos al JIT
        self.near_water_grid = calculate_water_effect(self.water_grid, rows, cols)

    def get_neighborhood(self, r: int, c: int) -> list[int]:
        """
        Obtiene los estados de las 8 celdas vecinas (vecindad de Moore).
        """
        # Llamamos a la versión optimizada y convertimos a lista para mantener API original
        # Nota: Numba retorna numpy array, el cast a list() satisface el tipado original
        return list(get_neighborhood(r, c, self.land))

    def get_transition_matrix(self, neighborhood_states: list[int], wind_direction: list[int], wind_intensity: float,
                              humidity: float, temperature: float, soil_moist: float, current_state: int,
                              local_dryness: float = 0.0) -> np.ndarray:
        """
        Calcula las probabilidades de transición.
        """
        # Convertimos inputs a tipos compatibles con numba si es necesario (list -> array)
        # Este método se mantiene por compatibilidad, pero update_step usa la versión rápida directa.
        neighbor_arr = np.array(neighborhood_states, dtype=np.int64)
        wind_arr = np.array(wind_direction, dtype=np.float64)

        return _jit_get_transition_probs(
            neighbor_arr, wind_arr, wind_intensity,
            humidity, temperature, soil_moist, current_state,
            local_dryness, self.grass_density
        )

    def update_step(self) -> np.ndarray:
        """
        Aplica el modelo de incendio basado en cadenas de Markov.
        """
        rows, cols = self.land.shape

        # Convertir dirección de viento a array para numba
        wind_arr = np.array(self.wind_direction, dtype=np.float64)

        # Delegar todo el bucle pesado a Numba
        self.land = _jit_update_loop(
            rows, cols, self.land, self.water_grid, self.dryness_grid,
            self.near_water_grid, wind_arr, self.wind_intensity,
            self.humidity, self.temperature, self.soil_moisture, self.grass_density
        )

        return self.land

    def compute_statistics(self) -> dict[str, float]:
        """
        Calcula las estadísticas de los estados.
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
        return (self.land == BURNING) | (self.land == BURNT)

    def apply_brush(self, r_center: int, c_center: int, brush_size: int, brush_type: str, value: float | None = None) -> None:
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
