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

#  Importamos y definimos variables locales de model_config para ser usados más rápidamente en este contexto
NEIGHBOR_COORDS_TUPLES = tuple(NEIGHBOR_RELATIVE_COORDS)
EMPTY_CONST = EMPTY
GRASS_CONST = GRASS
BURNING_CONST = BURNING
BURNT_CONST = BURNT
CHOICE_VALUES = np.array([EMPTY, GRASS, BURNING, BURNT], dtype=np.uint8)


@jit(nopython=True)
def _jit_get_neighborhood(r: int, c: int, land_grid: np.ndarray) -> np.ndarray:
    rows, cols = land_grid.shape
    neighbors = np.empty(8, dtype=np.uint8)
    idx = 0
    for dr, dc in NEIGHBOR_COORDS_TUPLES:
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
            neighbors[idx] = EMPTY_CONST
        else:
            neighbors[idx] = land_grid[nr, nc]
        idx += 1
    return neighbors


@jit(nopython=True)
def _jit_get_transition_matrix(neighborhood_states: np.ndarray, wind_direction_array: np.ndarray, wind_intensity: float,
                               humidity: float, temperature: float, soil_moist: float, current_state: int,
                               local_dryness: float = 0.0, grass_density: float = 0.0) -> np.ndarray:
    probs = np.zeros(4, dtype=np.float64)
    if current_state == EMPTY_CONST:
        probs[EMPTY_CONST] = 1.0
    elif current_state == GRASS_CONST:
        probs[GRASS_CONST] = 1.0
    elif current_state == BURNING_CONST:
        probs[BURNING_CONST] = 1.0
    elif current_state == BURNT_CONST:
        probs[BURNT_CONST] = 1.0

    burning_neighbors_count = 0
    for i in range(8):
        if neighborhood_states[i] == BURNING_CONST:
            burning_neighbors_count += 1

    if current_state == GRASS_CONST:
        total_wind_factor = 0.0
        if burning_neighbors_count > 0:
            wind_y_comp, wind_x_comp = wind_direction_array[0], wind_direction_array[1]
            norm_wind = math.hypot(wind_y_comp, wind_x_comp)
            for i in range(8):
                if neighborhood_states[i] == BURNING_CONST:
                    dr_n, dc_n = NEIGHBOR_COORDS_TUPLES[i]
                    norm_neighbor_vec = math.hypot(dr_n, dc_n)
                    if norm_neighbor_vec > 0 and norm_wind > 0:
                        dot_product = (-dr_n * wind_y_comp) + (-dc_n * wind_x_comp)
                        alignment = dot_product / (norm_neighbor_vec * norm_wind)
                        total_wind_factor += alignment

        humidity_reduction_factor = 1.0 - humidity
        soil_moisture_reduction_factor = 1.0 - (soil_moist * SOIL_MOISTURE_SENSITIVITY)
        soil_moisture_reduction_factor = max(0.0, soil_moisture_reduction_factor)
        temp_effect_on_spread = 0.0
        if burning_neighbors_count > 0:
            if temperature > TEMPERATURE_BASELINE:
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
    if probs_sum == 0.0:
        if current_state == EMPTY_CONST:
            probs[EMPTY_CONST] = 1.0
        elif current_state == GRASS_CONST:
            probs[GRASS_CONST] = 1.0
        elif current_state == BURNING_CONST:
            probs[BURNING_CONST] = 1.0
        elif current_state == BURNT_CONST:
            probs[BURNT_CONST] = 1.0
        else:
            probs[EMPTY_CONST] = 1.0
    else:
        probs = probs / probs_sum

    return probs


@jit(nopython=True)
def _jit_update_loop(rows: int, cols: int, land_grid: np.ndarray, water_grid: np.ndarray,
                     dryness_grid: np.ndarray, near_water_grid: np.ndarray,
                     wind_direction_array: np.ndarray, wind_intensity: float, humidity: float,
                     temperature: float, soil_moisture: float, grass_density: float) -> np.ndarray:
    new_grid = land_grid.copy()

    for r in range(rows):
        for c in range(cols):
            current_cell_state = EMPTY_CONST if water_grid[r, c] > 0 else land_grid[r, c]

            neighborhood_states = _jit_get_neighborhood(r, c, land_grid)

            try:
                local_dryness = float(dryness_grid[r, c])
            except Exception:
                local_dryness = 0.0

            near_water = near_water_grid[r, c]
            local_soil_moist = min(soil_moisture + (WATER_SOIL_MOISTURE_BONUS if near_water else 0.0), 1.0)

            transition_probs = _jit_get_transition_matrix(
                neighborhood_states=neighborhood_states,
                wind_direction_array=wind_direction_array,
                wind_intensity=wind_intensity,
                humidity=humidity,
                temperature=temperature,
                soil_moist=local_soil_moist,
                current_state=current_cell_state,
                local_dryness=local_dryness,
                grass_density=grass_density
            )

            if water_grid[r, c] > 0:
                new_grid[r, c] = EMPTY_CONST
            else:
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
        self.dryness_grid = np.full(self.grid_size, DEFAULT_GRASS_DRYNESS, dtype=float)
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

        _jit_update_loop(2, 2, np.array([[0, 0], [0, 0]], dtype=np.uint8),
                         np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=float),
                         np.zeros((2, 2), dtype=bool), self._wind_direction_array,
                         self.wind_intensity, self.humidity, self.temperature,
                         self.soil_moisture, self.grass_density)

    @property
    def wind_direction(self) -> list[int]:
        """
        Getter para la GUI.
        """
        return self._wind_direction_list

    @wind_direction.setter
    def wind_direction(self, new_direction: list[int]) -> None:
        """
        Setter para la GUI.
        """
        self._wind_direction_list = list(new_direction)
        self._wind_direction_array = np.array(new_direction, dtype=np.float64)

    def calculate_water_effect(self) -> None:
        self.near_water_grid = _jit_calculate_water_effect(self.water_grid, WATER_EFFECT_RADIUS)

    def update_step(self) -> np.ndarray:
        rows, cols = self.land.shape

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
