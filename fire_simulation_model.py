import numpy as np
import math
from numba import jit, prange
from config.model_config import (
    EMPTY, GRASS, BURNING, BURNT,
    DEFAULT_GRID_SIZE, DEFAULT_EMPTY_PROB, DEFAULT_GRASS_PROB, DEFAULT_GRASS_DRYNESS, DEFAULT_TEMPERATURE, DEFAULT_SOIL_MOISTURE,
    DEFAULT_WIND_DIRECTION, DEFAULT_WIND_INTENSITY, DEFAULT_HUMIDITY,
    P_G_B_BASE, F_G_B_NEIGHBOR, F_G_B_WIND, P_G_B_MAX,
    TEMPERATURE_BASELINE, TEMPERATURE_SENSITIVITY, SOIL_MOISTURE_SENSITIVITY,
    DRYNESS_SPREAD_MULTIPLIER, WATER_EFFECT_RADIUS, WATER_SOIL_MOISTURE_BONUS,
    NEIGHBOR_RELATIVE_COORDS, BASE_TRANSITIONS, DEFAULT_GRASS_DENSITY
)

"""
    Importamos y definimos variables locales de model_config para ser usados más rápidamente en este contexto
"""
NEIGHBOR_COORDS_TUPLES = tuple(NEIGHBOR_RELATIVE_COORDS)
EMPTY_CONST = EMPTY
GRASS_CONST = GRASS
BURNING_CONST = BURNING
BURNT_CONST = BURNT
CHOICE_VALUES = np.array([EMPTY, GRASS, BURNING, BURNT], dtype=np.uint8)
# Ya que NEIGHBOR_RELATIVE_COORDS no cambia durante la ejecución, podemos precalcularlo una vez y usarlo todas
NEIGHBOR_NORMS = tuple([math.hypot(dr, dc) for dr, dc in NEIGHBOR_RELATIVE_COORDS])


@jit(nopython=True)
def _jit_get_transition_probs(r: int, c: int, land_grid: np.ndarray, rows: int, cols: int, wind_direction_array: np.ndarray,
                              wind_intensity: float, humidity: float, temperature: float, soil_moist: float, current_state: int,
                              local_dryness: float, grass_density: float) -> np.ndarray:
    probs = BASE_TRANSITIONS[current_state].copy()

    if current_state == EMPTY_CONST or current_state == BURNT_CONST:
        return probs

    if current_state == GRASS_CONST:
        burning_neighbors_count = 0
        total_wind_factor = 0.0

        wind_y_comp = wind_direction_array[0]
        wind_x_comp = wind_direction_array[1]
        norm_wind = math.hypot(wind_y_comp, wind_x_comp)
        has_wind = norm_wind > 0

        for i in range(8):
            dr, dc = NEIGHBOR_COORDS_TUPLES[i]
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if land_grid[nr, nc] != BURNING_CONST:
                continue

            burning_neighbors_count += 1

            if has_wind:
                norm_neighbor = NEIGHBOR_NORMS[i]
                dot_product = (-dr * wind_y_comp) + (-dc * wind_x_comp)
                alignment = dot_product / (norm_neighbor * norm_wind)
                total_wind_factor += alignment

        humidity_reduction_factor = 1.0 - humidity
        soil_moisture_reduction_factor = max(0.0, 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY))
        temp_effect_on_spread = 0.0
        if burning_neighbors_count > 0 and temperature > TEMPERATURE_BASELINE:
            temp_effect_on_spread = (temperature - TEMPERATURE_BASELINE) * TEMPERATURE_SENSITIVITY

        ignition_prob_from_neighbors = F_G_B_NEIGHBOR * burning_neighbors_count
        ignition_prob_from_wind = total_wind_factor * wind_intensity * F_G_B_WIND
        dryness_scale = 1.0 + (max(0.0, min(100.0, local_dryness)) / 100.0) * DRYNESS_SPREAD_MULTIPLIER
        effective_spread_prob = (ignition_prob_from_neighbors + ignition_prob_from_wind + temp_effect_on_spread) \
                                * humidity_reduction_factor * soil_moisture_reduction_factor * dryness_scale
        final_burning_prob = min(max(P_G_B_BASE + effective_spread_prob, 0.0), P_G_B_MAX)
        probs[BURNING_CONST] = final_burning_prob
        probs[GRASS_CONST] = max(0.0, 1.0 - final_burning_prob)
        probs[EMPTY_CONST] = 0.0

    elif current_state == BURNING_CONST:
        expected_burn_steps = 2.0 + grass_density * (10.0 - 2.0)
        p_burn_to_burnt = 1.0 / max(2.0, expected_burn_steps)
        p_burning_to_burning = max(0.0, 1.0 - p_burn_to_burnt)
        probs[BURNING_CONST] = p_burning_to_burning
        probs[BURNT_CONST] = p_burn_to_burnt

    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        probs = BASE_TRANSITIONS[current_state].copy()

    return probs


# Agregamos parallel=True y prange para usar todos los núcleos del CPU
@jit(nopython=True, parallel=True)
def _jit_update_loop(rows: int, cols: int, land_grid: np.ndarray, water_grid: np.ndarray,
                     dryness_grid: np.ndarray, near_water_grid: np.ndarray,
                     wind_direction_array: np.ndarray, wind_intensity: float, humidity: float,
                     temperature: float, soil_moisture: float, grass_density: float) -> np.ndarray:
    new_grid = np.empty_like(land_grid)

    for r in prange(rows):
        for c in range(cols):
            if water_grid[r, c] > 0:
                new_grid[r, c] = EMPTY_CONST
                continue

            current_cell_state = land_grid[r, c]

            local_dryness = dryness_grid[r, c]
            near_water = near_water_grid[r, c]

            local_soil_moist = soil_moisture
            if near_water:
                local_soil_moist = min(soil_moisture + WATER_SOIL_MOISTURE_BONUS, 1.0)

            transition_probs = _jit_get_transition_probs(
                r, c, land_grid, rows, cols,
                wind_direction_array, wind_intensity,
                humidity, temperature, local_soil_moist,
                current_cell_state,
                local_dryness,
                grass_density
            )

            rand_val = np.random.rand()
            cumulative_prob = 0.0
            choice = EMPTY_CONST

            for i in range(4):
                cumulative_prob += transition_probs[i]
                if rand_val < cumulative_prob:
                    choice = CHOICE_VALUES[i]
                    break

            new_grid[r, c] = choice

    return new_grid


@jit(nopython=True)
def _jit_calculate_water_effect(water_grid: np.ndarray, radius: int) -> np.ndarray:
    rows, cols = water_grid.shape
    near_water_grid = np.zeros((rows, cols), dtype=np.bool_)

    for r in range(rows):
        for c in range(cols):
            r0 = max(0, r - radius)
            r1 = min(rows, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(cols, c + radius + 1)

            found_water = False
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    if water_grid[nr, nc] > 0:
                        found_water = True
                        break
                if found_water:
                    break

            if found_water:
                near_water_grid[r, c] = True

    return near_water_grid


class FireSimulationModel:
    def __init__(self) -> None:
        self.grid_size = DEFAULT_GRID_SIZE
        self.land = np.random.choice([EMPTY, GRASS], size=self.grid_size, p=[DEFAULT_EMPTY_PROB, DEFAULT_GRASS_PROB])
        self.dryness_grid = np.full(self.grid_size, DEFAULT_GRASS_DRYNESS, dtype=np.float64)
        self.water_grid = np.zeros(self.grid_size, dtype=np.uint8)

        self.temperature = DEFAULT_TEMPERATURE
        self.soil_moisture = DEFAULT_SOIL_MOISTURE
        self.wind_intensity = DEFAULT_WIND_INTENSITY
        self.humidity = DEFAULT_HUMIDITY
        self.grass_density = float(min(max(DEFAULT_GRASS_DENSITY, 0.0), 1.0))

        self._wind_direction_array = np.array(DEFAULT_WIND_DIRECTION, dtype=np.float64)
        self._wind_direction_list = list(DEFAULT_WIND_DIRECTION)

        self.near_water_grid = np.zeros(self.grid_size, dtype=bool)
        self.calculate_water_effect()

        # Se ejecuta el método principal una vez con datos falsos al inicio para que numba compile el código
        _jit_update_loop(2, 2, np.zeros((2, 2), dtype=np.uint8),
                         np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.float64),
                         np.zeros((2, 2), dtype=bool), self._wind_direction_array,
                         0.5, 0.5, 25.0, 0.5, 0.5)

    @property
    def wind_direction(self) -> list[int]:
        return self._wind_direction_list

    @wind_direction.setter
    def wind_direction(self, new_direction: list[int]) -> None:
        self._wind_direction_list = list(new_direction)
        self._wind_direction_array = np.array(new_direction, dtype=np.float64)

    def calculate_water_effect(self) -> None:
        self.near_water_grid = _jit_calculate_water_effect(self.water_grid, WATER_EFFECT_RADIUS)

    def update_step(self) -> np.ndarray:
        rows, cols = self.land.shape
        if self.dryness_grid.dtype != np.float64:
            self.dryness_grid = self.dryness_grid.astype(np.float64)

        self.land = _jit_update_loop(
            rows, cols,
            self.land, self.water_grid, self.dryness_grid,
            self.near_water_grid,
            self._wind_direction_array,
            self.wind_intensity, self.humidity, self.temperature,
            self.soil_moisture, self.grass_density
        )
        return self.land

    def compute_statistics(self) -> dict[str, float]:
        total_cells = self.land.size
        counts = np.bincount(self.land.flatten(), minlength=4)
        return {
            'empty': counts[EMPTY] / total_cells * 100,
            'grass': counts[GRASS] / total_cells * 100,
            'burning': counts[BURNING] / total_cells * 100,
            'burnt': counts[BURNT] / total_cells * 100
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
        self.dryness_grid = np.array(data["dryness_grid"], dtype=np.float64)
        self.water_grid = np.array(data["water_grid"], dtype=np.uint8)
        self.temperature = float(data["temperature"])
        self.soil_moisture = float(data["soil_moisture"])
        self.wind_direction = list(data["wind_direction"])
        self.wind_intensity = float(data["wind_intensity"])
        self.humidity = float(data["humidity"])
        self.grass_density = float(data["grass_density"])
        self.calculate_water_effect()
