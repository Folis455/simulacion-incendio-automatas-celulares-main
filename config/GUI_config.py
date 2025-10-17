FIGSIZE = (17, 8)
STATS_AXIS = dict(
    ylim=(0, 100),
    xlabel='Pasos de Simulación Reales',
    ylabel='Porcentaje de Celdas (%)',
    title='Evolución de Estados del Ecosistema',
)

# Rango sliders -> Valor Minimo, Valor Maximo, Tasa de Cambio
SLIDER_LIMITS = {
    "wind_x": (-10.0, 10.0, 0.5),
    "wind_y": (-10.0, 10.0, 0.5),
    "intensity": (0.0, 1.0, 0.05),
    "humidity": (0.0, 1.0, 0.05),
    "temperature": (0.0, 50.0, 0.01),
    "soil_moisture": (0.0, 1.0, 0.05),
    "speed": (1, 10, 1),
    "brush_size": (1, 25, 1),
    "brush_dryness": (0, 100, 1),
    "grass_density": (0, 100, 0.05),
}

COLORS = {
    "empty": (1.0, 1.0, 1.0),
    "green": (0.0, 0.5, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.4, 1.0),
    "black": (0.0, 0.0, 0.0),
}

# Animación
BASE_INTERVAL_MS = 5
