FIGSIZE = (17, 8)
STATS_AXIS = dict(
    xlim=(0, 50),
    ylim=(0, 100),
    xlabel='Pasos de Simulación Reales',
    ylabel='Porcentaje de Celdas (%)',
    title='Evolución de Estados del Ecosistema',
)

# Rango sliders -> Valor Minimo, Valor Maximo, Tasa de Cambio
SLIDER_LIMITS = {
    "wind_angle": (0, 360, 1),
    "wind_speed": (0.0, 15.0, 0.5),
    "humidity": (0.0, 1.0, 0.5),
    "temperature": (10.0, 50.0, 1),
    "soil_moisture": (0.0, 1.0, 0.5),
    "brush_size": (1, 25, 1),
    "brush_dryness": (0, 100, 1),
    "grass_density": (0, 100, 0.5),
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
BASE_INTERVAL_MS = 500
