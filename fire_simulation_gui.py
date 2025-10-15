import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog
from fire_simulation_model import FireSimulationModel, EMPTY, GRASS, BURNING, BURNT
from config.GUI_config import FIGSIZE, SLIDER_LIMITS, COLORS, BASE_INTERVAL_MS
from config.model_config import DEFAULT_GRID_SIZE


class FireSimulationGUI:
    """
    Interfaz gráfica para la simulación de incendios de pastizales.
    """

    def __init__(self, simulation_steps=50):
        """
        Inicializa la interfaz gráfica.
        
        Args:
            simulation_steps (int): Número de pasos de simulación
        """
        self.grid_size = DEFAULT_GRID_SIZE
        self.simulation_steps = simulation_steps
        self.model = FireSimulationModel()
        self.grass_density = self.model.grass_density

        # Variables de control de simulación
        self.simulation_started = False
        self.simulation_paused = False
        self.ani = None
        self.stats_history = []
        self.simulation_speed_multiplier = 1

        # Variables de pincel
        self.brush_size_cells = 1
        self.brush_dryness_value = 50
        self.painting_mode = 'fire'  # 'fire', 'dryness', 'water', 'grass', 'empty'
        self.show_water_overlay = False

        # Variables de interacción
        self.current_highlight_patch = None
        self.is_mouse_button_down = False
        self.brush_info_text = None

        # Configurar la interfaz
        self._setup_ui()
        self._setup_controls()
        self._setup_event_handlers()

        # Mostrar texto inicial del pincel
        self.update_brush_text()

    def _setup_ui(self):
        """Configura la interfaz de usuario básica."""
        # Configuración de la figura
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

        # Panel de simulación
        self.im = self.ax1.imshow(self._build_display_image(), animated=True)
        self.ax1.set_title("Simulación de Incendio (Autómata Celular Estocástico)")
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        # Panel de estadísticas
        colors = ["white", "green", "red", "black"]
        self.line_empty, = self.ax2.plot([], [], color='lightgray', linestyle='-', label='Vacío (%)')
        self.line_grass, = self.ax2.plot([], [], color=colors[GRASS], linestyle='-', label='Pasto (%)')
        self.line_burning, = self.ax2.plot([], [], color=colors[BURNING], linestyle='-', label='Quemando (%)')
        self.line_burnt, = self.ax2.plot([], [], color=colors[BURNT], linestyle='-', label='Quemado (%)')

        self.ax2.set_xlim(0, self.simulation_steps)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_xlabel('Pasos de Simulación Reales')
        self.ax2.set_ylabel('Porcentaje de Celdas (%)')
        self.ax2.legend()
        self.ax2.set_title("Evolución de Estados del Ecosistema")
        self.ax2.grid(True)

        # Ajustar layout para hacer espacio a los controles
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.45)

    def _setup_controls(self):
        """Configura los sliders y botones de control."""
        # Crear ejes para los sliders
        ax_wind_x = plt.axes([0.15, 0.40, 0.7, 0.025])
        ax_wind_y = plt.axes([0.15, 0.35, 0.7, 0.025])
        ax_intensity = plt.axes([0.15, 0.30, 0.7, 0.025])
        ax_humidity = plt.axes([0.15, 0.25, 0.7, 0.025])
        ax_temperature = plt.axes([0.15, 0.20, 0.7, 0.025])
        ax_soil_moisture = plt.axes([0.15, 0.15, 0.7, 0.025])
        ax_speed = plt.axes([0.15, 0.10, 0.7, 0.025])
        ax_brush_size = plt.axes([0.15, 0.075, 0.22, 0.025])
        ax_brush_dryness = plt.axes([0.39, 0.075, 0.22, 0.025])
        ax_grass_density = plt.axes([0.63, 0.075, 0.22, 0.025])

        # Crear sliders
        self.slider_wind_x = Slider(
            ax_wind_x, label='Viento X',
            valmin=SLIDER_LIMITS["wind_x"][0], valmax=SLIDER_LIMITS["wind_x"][1],
            valinit=self.model.wind_direction[1], valstep=SLIDER_LIMITS["wind_x"][2]
        )
        self.slider_wind_y = Slider(
            ax_wind_y, label='Viento Y',
            valmin=SLIDER_LIMITS["wind_y"][0], valmax=SLIDER_LIMITS["wind_y"][1],
            valinit=self.model.wind_direction[0], valstep=SLIDER_LIMITS["wind_y"][2]
        )
        self.slider_intensity = Slider(
            ax_intensity, label='Intensidad Viento',
            valmin=SLIDER_LIMITS["intensity"][0], valmax=SLIDER_LIMITS["intensity"][1],
            valinit=self.model.wind_intensity, valstep=SLIDER_LIMITS["intensity"][2]
        )
        self.slider_humidity = Slider(
            ax_humidity, label='Humedad Aire',
            valmin=SLIDER_LIMITS["humidity"][0], valmax=SLIDER_LIMITS["humidity"][1],
            valinit=self.model.humidity, valstep=SLIDER_LIMITS["humidity"][2]
        )
        self.slider_temperature = Slider(
            ax_temperature, label='Temperatura (°C)',
            valmin=SLIDER_LIMITS["temperature"][0], valmax=SLIDER_LIMITS["temperature"][1],
            valinit=self.model.temperature, valstep=SLIDER_LIMITS["temperature"][2]
        )
        self.slider_soil_moisture = Slider(
            ax_soil_moisture, label='Humedad Suelo',
            valmin=SLIDER_LIMITS["soil_moisture"][0], valmax=SLIDER_LIMITS["soil_moisture"][1],
            valinit=self.model.soil_moisture, valstep=SLIDER_LIMITS["soil_moisture"][2]
        )
        self.slider_speed = Slider(
            ax_speed, label='Velocidad (Pasos/Cuadro)',
            valmin=SLIDER_LIMITS["speed"][0], valmax=SLIDER_LIMITS["speed"][1],
            valinit=self.simulation_speed_multiplier, valstep=SLIDER_LIMITS["speed"][2]
        )
        self.slider_brush_size = Slider(
            ax_brush_size, label='Tamaño Pincel (celdas)',
            valmin=SLIDER_LIMITS["brush_size"][0], valmax=SLIDER_LIMITS["brush_size"][1],
            valinit=self.brush_size_cells, valstep=SLIDER_LIMITS["brush_size"][2]
        )
        self.slider_brush_dryness = Slider(
            ax_brush_dryness, label='Sequedad [0-100]',
            valmin=SLIDER_LIMITS["brush_dryness"][0], valmax=SLIDER_LIMITS["brush_dryness"][1],
            valinit=self.brush_dryness_value, valstep=SLIDER_LIMITS["brush_dryness"][2]
        )
        self.slider_grass_density = Slider(
            ax_grass_density, label='Densidad Pasto',
            valmin=SLIDER_LIMITS["grass_density"][0], valmax=SLIDER_LIMITS["grass_density"][1],
            valinit=self.grass_density, valstep=SLIDER_LIMITS["grass_density"][2]
        )

        # Conectar sliders a funciones de actualización
        self.slider_wind_x.on_changed(self._update_wind_x)
        self.slider_wind_y.on_changed(self._update_wind_y)
        self.slider_intensity.on_changed(self._update_intensity)
        self.slider_humidity.on_changed(self._update_humidity)
        self.slider_temperature.on_changed(self._update_temperature)
        self.slider_soil_moisture.on_changed(self._update_soil_moisture)
        self.slider_speed.on_changed(self._update_speed)
        self.slider_brush_size.on_changed(self._update_brush_size)
        self.slider_brush_dryness.on_changed(self._update_brush_dryness)
        self.slider_grass_density.on_changed(self._update_grass_density)

        # Botones de control
        button_color = 'lightgoldenrodyellow'
        ax_button_main = plt.axes([0.25, 0.02, 0.2, 0.035])
        self.button_main = Button(ax_button_main, 'Iniciar Simulación',
                                  color=button_color, hovercolor='0.975')

        ax_button_finalize = plt.axes([0.55, 0.02, 0.2, 0.035])
        self.button_finalize = Button(ax_button_finalize, 'Finalizar Simulación',
                                      color=button_color, hovercolor='0.975')
        self.button_finalize.ax.set_visible(False)

        self.button_main.on_clicked(self._main_button_callback)
        self.button_finalize.on_clicked(self._finalize_button_callback)

        ax_button_save = plt.axes([0.05, 0.02, 0.15, 0.035])
        self.button_save = Button(ax_button_save, 'Guardar Config.')
        self.button_save.on_clicked(self._save_button_callback)

        ax_button_load = plt.axes([0.75, 0.02, 0.15, 0.035])
        self.button_load = Button(ax_button_load, 'Cargar Config.')
        self.button_load.on_clicked(self._load_button_callback)

    def _setup_event_handlers(self):
        """Configura los manejadores de eventos."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_display_image(self):
        """
        Construye la imagen de visualización con colores basados en estados y sequedad.
        
        Returns:
            np.array: Imagen RGB de la cuadrícula
        """
        h, w = self.model.land.shape
        img = np.zeros((h, w, 3), dtype=float)

        # Colores base
        color_empty = np.array(COLORS["empty"])
        color_green = np.array(COLORS["green"])
        color_yellow = np.array(COLORS["yellow"])
        color_red = np.array(COLORS["red"])
        color_blue = np.array(COLORS["blue"])
        color_black = np.array(COLORS["black"])

        # Máscaras por estado
        mask_empty = (self.model.land == EMPTY)
        mask_grass = (self.model.land == GRASS)
        mask_burning = (self.model.land == BURNING)
        mask_burnt = (self.model.land == BURNT)

        # Aplicar colores base
        img[mask_empty] = color_empty
        img[mask_burning] = color_red
        img[mask_burnt] = color_black

        # Pasto con degradado según sequedad (0 -> verde, 100 -> amarillo)
        dryness_norm = np.clip(self.model.dryness_grid / 100.0, 0.0, 1.0)
        grass_idx = np.where(mask_grass)
        if grass_idx[0].size > 0:
            w_yellow = np.power(dryness_norm[grass_idx], 1.5)
            w_green = 1.0 - w_yellow
            img[grass_idx] = (w_green[:, None] * color_green) + (w_yellow[:, None] * color_yellow)

        # Celdas de agua: azules con overlay opcional
        if np.any(self.model.water_grid):
            water_mask = (self.model.water_grid > 0)
            img[water_mask] = color_blue

            if self.show_water_overlay:
                # Mostrar área de influencia del agua
                from fire_simulation_model import WATER_EFFECT_RADIUS
                pad = WATER_EFFECT_RADIUS
                padded = np.pad(self.model.water_grid, pad_width=pad, mode='constant', constant_values=0)
                near = np.zeros_like(self.model.water_grid, dtype=bool)
                for dr in range(-pad, pad + 1):
                    for dc in range(-pad, pad + 1):
                        near |= padded[pad + dr:pad + dr + self.model.water_grid.shape[0],
                                pad + dc:pad + dc + self.model.water_grid.shape[1]] > 0
                overlay_mask = near & (~water_mask)
                if np.any(overlay_mask):
                    cyan = np.array([0.6, 0.9, 1.0])
                    alpha = 0.25
                    img[overlay_mask] = (1 - alpha) * img[overlay_mask] + alpha * cyan

        return img

    def update_brush_text(self):
        """Actualiza el texto de información del pincel."""
        status = f"Modo: {self.painting_mode.upper()} | Pincel: {self.brush_size_cells} | Sequedad: {int(self.brush_dryness_value)}"
        if self.brush_info_text is None:
            self.brush_info_text = self.fig.text(0.05, 0.94, status)
        else:
            self.brush_info_text.set_text(status)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Maneja eventos de clic del ratón."""
        if event.inaxes != self.ax1:
            return

        if event.button == 1:  # Botón izquierdo
            self.is_mouse_button_down = True
            c, r = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                self._apply_brush_at(r, c)

    def _on_release(self, event):
        """Maneja eventos de liberación del ratón."""
        if event.button == 1:
            self.is_mouse_button_down = False

    def _on_hover(self, event):
        """Maneja eventos de movimiento del ratón."""
        # Manejar resaltado
        if event.inaxes == self.ax1:
            c_hover, r_hover = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= r_hover < self.grid_size[0] and 0 <= c_hover < self.grid_size[1]:
                if self.current_highlight_patch:
                    self.current_highlight_patch.remove()
                self.current_highlight_patch = patches.Rectangle(
                    (c_hover - 0.5, r_hover - 0.5), 1, 1,
                    linewidth=1.5, edgecolor='yellow', facecolor='none', zorder=10
                )
                self.ax1.add_patch(self.current_highlight_patch)
            else:
                if self.current_highlight_patch:
                    self.current_highlight_patch.remove()
                    self.current_highlight_patch = None
        else:
            if self.current_highlight_patch:
                self.current_highlight_patch.remove()
                self.current_highlight_patch = None

        # Manejar pintado mientras se arrastra
        if self.is_mouse_button_down and event.inaxes == self.ax1:
            c_paint, r_paint = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= r_paint < self.grid_size[0] and 0 <= c_paint < self.grid_size[1]:
                self._apply_brush_at(r_paint, c_paint)

        if (event.inaxes == self.ax1 and (self.is_mouse_button_down or self.current_highlight_patch is not None)) or \
                (event.inaxes != self.ax1 and self.current_highlight_patch is None):
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Maneja eventos de teclado."""
        if event.key == 'm':
            # Rotar modo de pincel
            modes = ['fire', 'dryness', 'water', 'grass', 'empty']
            current_idx = modes.index(self.painting_mode)
            self.painting_mode = modes[(current_idx + 1) % len(modes)]
            self.update_brush_text()
        elif event.key == '.':
            self.brush_size_cells = min(25, self.brush_size_cells + 1)
            self.update_brush_text()
        elif event.key == ',':
            self.brush_size_cells = max(1, self.brush_size_cells - 1)
            self.update_brush_text()
        elif event.key == '|':
            self.brush_dryness_value = 0
            self.update_brush_text()
        elif event.key in '1234567890':
            self.brush_dryness_value = int(event.key) * 10 if event.key != '0' else 100
            self.update_brush_text()
        elif event.key == '{':
            self.brush_dryness_value = max(0, self.brush_dryness_value - 5)
            self.update_brush_text()
        elif event.key == '}':
            self.brush_dryness_value = min(100, self.brush_dryness_value + 5)
            self.update_brush_text()
        elif event.key == 'h':
            self.show_water_overlay = not self.show_water_overlay
            self.im.set_array(self._build_display_image())
            self.fig.canvas.draw_idle()

    def _apply_brush_at(self, r, c):
        """Aplica el pincel en la posición especificada."""
        if self.painting_mode == 'dryness':
            self.model.apply_brush(r, c, self.brush_size_cells, 'dryness', self.brush_dryness_value)
        else:
            self.model.apply_brush(r, c, self.brush_size_cells, self.painting_mode)

        self.im.set_array(self._build_display_image())

    # Funciones de actualización de sliders
    def _update_wind_x(self, val):
        self.model.wind_direction[1] = val

    def _update_wind_y(self, val):
        self.model.wind_direction[0] = val

    def _update_intensity(self, val):
        self.model.wind_intensity = val

    def _update_humidity(self, val):
        self.model.humidity = val

    def _update_temperature(self, val):
        self.model.temperature = val

    def _update_soil_moisture(self, val):
        self.model.soil_moisture = val

    def _update_speed(self, val):
        self.simulation_speed_multiplier = int(val)
        # Actualizar interval de animación si está corriendo
        if self.simulation_started and not self.simulation_paused and self.ani:
            base_interval = BASE_INTERVAL_MS
            dynamic_interval = max(10, base_interval // self.simulation_speed_multiplier)
            self.ani.event_source.interval = dynamic_interval

    def _update_brush_size(self, val):
        self.brush_size_cells = int(val)
        self.update_brush_text()

    def _update_brush_dryness(self, val):
        self.brush_dryness_value = int(val)
        self.update_brush_text()

    def _update_grass_density(self, val):
        self.grass_density = float(val)
        self.model.grass_density = self.grass_density

    def _main_button_callback(self, event):
        """Maneja el botón principal (Iniciar/Pausar/Reanudar)."""
        if not self.simulation_started:
            # Iniciar simulación
            self.simulation_started = True
            self.simulation_paused = False

            self.button_main.label.set_text("Pausar")
            self.button_main.ax.set_facecolor('red')
            self.button_finalize.ax.set_visible(False)

            # Verificar si hay fuego, si no, iniciarlo en el centro
            if BURNING not in self.model.land:
                center_r, center_c = self.grid_size[0] // 2, self.grid_size[1] // 2
                self.model.apply_brush(center_r, center_c, 1, 'fire')
                self.im.set_array(self._build_display_image())

            self.stats_history = []
            # Calcular interval dinámico basado en velocidad para verdadero fast-forward
            base_interval = BASE_INTERVAL_MS
            dynamic_interval = max(10, base_interval // self.simulation_speed_multiplier)
            self.ani = animation.FuncAnimation(self.fig, self._update_animation,
                                               frames=self.simulation_steps, interval=dynamic_interval, blit=False)
            self.fig.canvas.draw_idle()

        elif self.simulation_started and not self.simulation_paused:
            # Pausar simulación
            self.simulation_paused = True
            if self.ani:
                self.ani.event_source.stop()
            self.button_main.label.set_text("Reanudar")
            self.button_main.ax.set_facecolor('lightgreen')
            self.button_finalize.ax.set_visible(True)
            self.fig.canvas.draw_idle()

        elif self.simulation_started and self.simulation_paused:
            # Reanudar simulación
            self.simulation_paused = False
            if self.ani:
                self.ani.event_source.start()
            self.button_main.label.set_text("Pausar")
            self.button_main.ax.set_facecolor('red')
            self.button_finalize.ax.set_visible(False)
            self.fig.canvas.draw_idle()

    def _finalize_button_callback(self, event):
        """Maneja el botón de finalizar simulación."""
        if not self.simulation_started or not self.simulation_paused:
            return

        if self.ani:
            self.ani.event_source.stop()
        self._reset_simulation_state()

    def _reset_simulation_state(self):
        """Resetea la simulación a su estado inicial."""
        self.simulation_started = False
        self.simulation_paused = False

        self.model = FireSimulationModel()
        self.im.set_array(self._build_display_image())

        self.stats_history = []
        x_data = []
        empty_data = []
        grass_data = []
        burning_data = []
        burnt_data = []

        self.line_empty.set_data(x_data, empty_data)
        self.line_grass.set_data(x_data, grass_data)
        self.line_burning.set_data(x_data, burning_data)
        self.line_burnt.set_data(x_data, burnt_data)
        self.ax2.set_xlim(0, self.simulation_steps)

        self.button_main.label.set_text("Iniciar Simulación")
        self.button_main.ax.set_facecolor('lightgoldenrodyellow')
        self.button_finalize.ax.set_visible(False)

        self.fig.canvas.draw_idle()
        self.update_brush_text()

    def _update_animation(self, frame):
        """Actualiza la animación en cada frame."""
        if not self.simulation_started or self.simulation_paused:
            return [self.im, self.line_empty, self.line_grass, self.line_burning, self.line_burnt]

        # Ejecutar UN solo paso por frame (velocidad controlada por interval)
        self.model.update_step()
        current_stats = self.model.compute_statistics()
        self.stats_history.append(current_stats)

        self.im.set_array(self._build_display_image())

        # Actualizar gráfico de estadísticas
        x_data = list(range(len(self.stats_history)))
        self.line_empty.set_data(x_data, [s['empty'] for s in self.stats_history])
        self.line_grass.set_data(x_data, [s['grass'] for s in self.stats_history])
        self.line_burning.set_data(x_data, [s['burning'] for s in self.stats_history])
        self.line_burnt.set_data(x_data, [s['burnt'] for s in self.stats_history])

        if x_data:
            # Ajustar rango del gráfico basado en pasos reales ejecutados
            self.ax2.set_xlim(0, max(self.simulation_steps, len(x_data)) + 1)

        return [self.im, self.line_empty, self.line_grass, self.line_burning, self.line_burnt]

    def show(self):
        """Muestra la interfaz gráfica."""
        plt.show()

    # --------- Guardar y cargar configuración ----------
    def _save_button_callback(self, event):
        self.save_configs_to_file()

    def _load_button_callback(self, event):
        self.load_configs_from_file()

    def save_configs_to_file(self):
        """Guarda el estado del modelo en un archivo NPZ comprimido."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy compressed", "*.npz")],
            title="Guardar configuración de simulación"
        )
        if file_path:
            np.savez_compressed(
                file_path,
                grid_size=np.array(self.model.grid_size),
                land=self.model.land,
                dryness_grid=self.model.dryness_grid,
                water_grid=self.model.water_grid,
                temperature=self.model.temperature,
                soil_moisture=self.model.soil_moisture,
                wind_direction=np.array(self.model.wind_direction),
                wind_intensity=self.model.wind_intensity,
                humidity=self.model.humidity,
                grass_density=self.model.grass_density
            )

    def load_configs_from_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy compressed", "*.npz")],
            title="Cargar configuración de simulación"
        )
        if file_path:
            with np.load(file_path) as data:
                self.model.import_state(data)
                # Actualizar sliders con los valores cargados
                self.slider_wind_x.set_val(self.model.wind_direction[1])
                self.slider_wind_y.set_val(self.model.wind_direction[0])
                self.slider_intensity.set_val(self.model.wind_intensity)
                self.slider_humidity.set_val(self.model.humidity)
                self.slider_temperature.set_val(self.model.temperature)
                self.slider_soil_moisture.set_val(self.model.soil_moisture)
                self.slider_speed.set_val(self.simulation_speed_multiplier)
                self.slider_brush_size.set_val(self.brush_size_cells)
                self.slider_brush_dryness.set_val(self.brush_dryness_value)
                self.slider_grass_density.set_val(self.grass_density)

                # Para que se cargue el grid importado
                if not self.simulation_started:
                    self.simulation_started = True
                self._update_animation(frame=0)
                self.fig.canvas.draw_idle()
