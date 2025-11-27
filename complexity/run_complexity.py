import time
from fire_simulation_model import FireSimulationModel
from config.model_config import DEFAULT_GRID_SIZE

model = FireSimulationModel()
model.apply_brush(DEFAULT_GRID_SIZE[0] // 2, DEFAULT_GRID_SIZE[1] // 2, 5, 'fire')

NUM_STEPS = 40

print(f"Iniciando simulación: {DEFAULT_GRID_SIZE[0]}x{DEFAULT_GRID_SIZE[1]} celdas, {NUM_STEPS} pasos...")
start_time = time.time()

for step in range(NUM_STEPS):
    model.update_step()

end_time = time.time()
total_time = end_time - start_time

print(f"Simulación completada.")
print(f"Tiempo total: {total_time:.4f} segundos")
print(f"Tiempo promedio por paso: {(total_time / NUM_STEPS):.6f} segundos")
