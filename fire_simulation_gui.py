from fire_simulation_model import FireSimulationModel, EMPTY, GRASS, BURNING, BURNT
from config.GUI_config import FIGSIZE, SLIDER_LIMITS, COLORS, BASE_INTERVAL_MS, STATS_AXIS
from config.model_config import DEFAULT_GRID_SIZE

import numpy as np
from numpy.lib._stride_tricks_impl import as_strided
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog


class FireSimulationGUI:
    """
    Interfaz gráfica para la simulación de incendios de pastizales.
    """

    def __init__(self):
        """
        Inicializa la interfaz gráfica.
        """
        self.grid_size = DEFAULT_GRID_SIZE
        self.model = FireSimulationModel()
        self.grass_density = self.model.grass_density

        model_y, model_x = self.model.wind_direction
        model_intensity = self.model.wind_intensity
        self.wind_angle = np.rad2deg(np.arctan2(-model_x, model_y))
        if self.wind_angle < 0:
            self.wind_angle += 360
        max_speed = SLIDER_LIMITS["wind_speed"][1]
        self.wind_speed = model_intensity * max_speed

        self.simulation_paused = True
        self.current_speed_mode = 'pause'  # 'pause', 'play', 'turbo'
        self.ani = None
        self.stats_history = []

        # Variables de pincel
        self.brush_size_cells = 1
        self.brush_dryness_value = 50
        self.painting_mode = 'fire'  # 'fire', 'dryness', 'water', 'grass', 'empty'
        self.show_water_overlay = False

        # Variables de interacción
        self.current_highlight_patch = None
        self.is_mouse_button_down = False
        self.brush_info_text = None

        self.active_pause_button_color = 'lightcoral'
        self.active_button_color = 'lightgreen'
        self.inactive_button_color = 'lightgoldenrodyellow'

        self._setup_ui()
        self._setup_controls()
        self._setup_event_handlers()
        self._calculate_and_set_wind()

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

        compass_props = dict(
            fontsize=8,
            fontweight='bold',
            color='black',
            transform=self.ax1.transAxes
        )
        self.ax1.text(0.5, 0.95, '0° (N)', ha='center', va='top', **compass_props)
        self.ax1.text(0.5, 0.05, '180° (S)', ha='center', va='bottom', **compass_props)
        self.ax1.text(0.95, 0.5, '90° (E)', ha='right', va='center', **compass_props)
        self.ax1.text(0.05, 0.5, '270° (O)', ha='left', va='center', **compass_props)

        # Panel de estadísticas (Ahora usa STATS_AXIS)
        colors = ["white", "green", "red", "black"]
        self.line_empty, = self.ax2.plot([], [], color='lightgray', linestyle='-', label='Vacío (%)')
        self.line_grass, = self.ax2.plot([], [], color=colors[GRASS], linestyle='-', label='Pasto (%)')
        self.line_burning, = self.ax2.plot([], [], color=colors[BURNING], linestyle='-', label='Quemando (%)')
        self.line_burnt, = self.ax2.plot([], [], color=colors[BURNT], linestyle='-', label='Quemado (%)')

        self.ax2.set_xlim(STATS_AXIS['xlim'])
        self.ax2.set_ylim(STATS_AXIS['ylim'])
        self.ax2.set_xlabel(STATS_AXIS['xlabel'])
        self.ax2.set_ylabel(STATS_AXIS['ylabel'])
        self.ax2.set_title(STATS_AXIS['title'])

        self.ax2.legend()
        self.ax2.grid(True)

        # Ajustar layout para hacer espacio a los controles
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.45)

        # Se crea la animación y se pausa inmediatamente.
        self.ani = self._create_animation(BASE_INTERVAL_MS)
        self.ani.event_source.stop()

    def _create_animation(self, interval_ms):
        interval_ms = max(10, int(interval_ms))

        return animation.FuncAnimation(
            self.fig, self._update_animation,
            frames=None,
            interval=interval_ms,
            blit=False
        )

    def _setup_controls(self):
        """Configura los sliders y botones de control."""
        # Crear ejes para los sliders
        ax_wind_angle = plt.axes([0.15, 0.40, 0.7, 0.025])
        ax_wind_speed = plt.axes([0.15, 0.35, 0.7, 0.025])
        ax_humidity = plt.axes([0.15, 0.30, 0.7, 0.025])
        ax_temperature = plt.axes([0.15, 0.25, 0.7, 0.025])
        ax_soil_moisture = plt.axes([0.15, 0.20, 0.7, 0.025])
        ax_brush_size = plt.axes([0.15, 0.15, 0.7, 0.025])
        ax_brush_dryness = plt.axes([0.15, 0.10, 0.7, 0.025])
        ax_grass_density = plt.axes([0.15, 0.05, 0.7, 0.025])

        # Crear sliders
        self.slider_wind_angle = Slider(
            ax_wind_angle, label='Dirección Viento (°)',
            valmin=SLIDER_LIMITS["wind_angle"][0], valmax=SLIDER_LIMITS["wind_angle"][1],
            valinit=self.wind_angle, valstep=SLIDER_LIMITS["wind_angle"][2]
        )
        self.slider_wind_speed = Slider(
            ax_wind_speed, label='Velocidad Viento',
            valmin=SLIDER_LIMITS["wind_speed"][0], valmax=SLIDER_LIMITS["wind_speed"][1],
            valinit=self.wind_speed, valstep=SLIDER_LIMITS["wind_speed"][2]
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
        self.slider_wind_angle.on_changed(self._update_wind_angle)
        self.slider_wind_speed.on_changed(self._update_wind_speed)
        self.slider_humidity.on_changed(self._update_humidity)
        self.slider_temperature.on_changed(self._update_temperature)
        self.slider_soil_moisture.on_changed(self._update_soil_moisture)
        self.slider_brush_size.on_changed(self._update_brush_size)
        self.slider_brush_dryness.on_changed(self._update_brush_dryness)
        self.slider_grass_density.on_changed(self._update_grass_density)

        # --- Botones de Control ---
        ax_button_pause = plt.axes([0.30, 0.02, 0.1, 0.035])
        ax_button_play = plt.axes([0.41, 0.02, 0.1, 0.035])
        ax_button_turbo = plt.axes([0.52, 0.02, 0.1, 0.035])
        ax_button_reset = plt.axes([0.63, 0.02, 0.1, 0.035])

        self.button_pause = Button(ax_button_pause, 'Pausa', color=self.inactive_button_color, hovercolor='0.975')
        self.button_play = Button(ax_button_play, 'Play', color=self.inactive_button_color, hovercolor='0.975')
        self.button_turbo = Button(ax_button_turbo, 'Turbo', color=self.inactive_button_color, hovercolor='0.975')
        self.button_reset = Button(ax_button_reset, 'Reset', color=self.inactive_button_color, hovercolor='0.975')

        self.button_pause.on_clicked(self._on_pause_click)
        self.button_play.on_clicked(self._on_play_click)
        self.button_turbo.on_clicked(self._on_turbo_click)
        self.button_reset.on_clicked(self._on_reset_click)
        # ----------------------------------

        ax_button_save = plt.axes([0.05, 0.02, 0.15, 0.035])
        self.button_save = Button(ax_button_save, 'Guardar Config.')
        self.button_save.on_clicked(self._save_button_callback)

        ax_button_load = plt.axes([0.75, 0.02, 0.15, 0.035])
        self.button_load = Button(ax_button_load, 'Cargar Config.')
        self.button_load.on_clicked(self._load_button_callback)

        self._update_speed_buttons()

    def _setup_event_handlers(self):
        """Configura los manejadores de eventos."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_display_image(self):
        """Construye la imagen de visualización"""
        h, w = self.model.land.shape
        img = np.zeros((h, w, 3), dtype=float)

        colors = {k: np.array(v) for k, v in COLORS.items()}
        mask_empty = (self.model.land == EMPTY)
        mask_grass = (self.model.land == GRASS)
        mask_burning = (self.model.land == BURNING)
        mask_burnt = (self.model.land == BURNT)

        img[mask_empty] = colors["empty"]
        img[mask_burning] = colors["red"]
        img[mask_burnt] = colors["black"]

        if np.any(mask_grass):
            dryness_norm = np.clip(self.model.dryness_grid / 100.0, 0.0, 1.0)
            w_yellow = np.power(dryness_norm[mask_grass], 1.5)
            w_green = 1.0 - w_yellow
            img[mask_grass] = w_green[:, None] * colors["green"] + w_yellow[:, None] * colors["yellow"]

        water_mask = self.model.water_grid > 0
        if np.any(water_mask):
            img[water_mask] = colors["blue"]
            if self.show_water_overlay:
                from fire_simulation_model import WATER_EFFECT_RADIUS
                pad = WATER_EFFECT_RADIUS
                padded = np.pad(water_mask, pad, mode='constant', constant_values=False)
                shape = (h, w, 2 * pad + 1, 2 * pad + 1)
                strides = padded.strides + padded.strides
                windows = as_strided(padded, shape=shape, strides=strides)
                overlay_mask = np.any(windows, axis=(2, 3))
                overlay_mask &= ~water_mask
                if np.any(overlay_mask):
                    alpha = 0.25
                    cyan = colors["cyan"]
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
        if event.inaxes != self.ax1:
            if self.current_highlight_patch:
                self.current_highlight_patch.remove()
                self.current_highlight_patch = None
            return

        if not self.simulation_paused:
            return

        c_hover, r_hover = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= r_hover < self.grid_size[0] and 0 <= c_hover < self.grid_size[1]:
            if self.current_highlight_patch:
                self.current_highlight_patch.set_xy((c_hover - 0.5, r_hover - 0.5))
            else:
                self.current_highlight_patch = patches.Rectangle(
                    (c_hover - 0.5, r_hover - 0.5), 1, 1,
                    linewidth=1.5, edgecolor='yellow', facecolor='none', zorder=10
                )
                self.ax1.add_patch(self.current_highlight_patch)
            self.fig.canvas.draw_idle()
        else:
            if self.current_highlight_patch:
                self.current_highlight_patch.remove()
                self.current_highlight_patch = None
        if self.is_mouse_button_down:
            self._on_click(event)

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

    # --------- Funciones de actualización de sliders ----------
    def _update_humidity(self, val):
        self.model.humidity = val

    def _update_temperature(self, val):
        self.model.temperature = val

    def _update_soil_moisture(self, val):
        self.model.soil_moisture = val

    def _update_brush_size(self, val):
        self.brush_size_cells = int(val)
        self.update_brush_text()

    def _update_brush_dryness(self, val):
        self.brush_dryness_value = int(val)
        self.update_brush_text()

    def _update_grass_density(self, val):
        self.grass_density = float(val)
        self.model.grass_density = self.grass_density

    def _update_speed_buttons(self):
        """Actualiza el color de los botones de control."""
        self.button_pause.color = self.active_pause_button_color if self.current_speed_mode == 'pause' else self.inactive_button_color
        self.button_pause.ax.set_facecolor(self.button_pause.color)

        self.button_play.color = self.active_button_color if self.current_speed_mode == 'play' else self.inactive_button_color
        self.button_play.ax.set_facecolor(self.button_play.color)

        self.button_turbo.color = self.active_button_color if self.current_speed_mode == 'turbo' else self.inactive_button_color
        self.button_turbo.ax.set_facecolor(self.button_turbo.color)

        if self.simulation_paused:
            self.button_reset.set_active(True)
            self.button_reset.color = self.inactive_button_color
            self.button_reset.ax.set_facecolor(self.inactive_button_color)

            self.button_save.set_active(True)
            self.button_save.color = 'lightgoldenrodyellow'
            self.button_save.ax.set_facecolor('lightgoldenrodyellow')

            self.button_load.set_active(True)
            self.button_load.color = 'lightgoldenrodyellow'
            self.button_load.ax.set_facecolor('lightgoldenrodyellow')

        else:
            # Deshabilitar Reset, Guardar y Cargar
            disabled_color = '0.85'  # Un color gris claro

            self.button_reset.set_active(False)
            self.button_reset.color = disabled_color
            self.button_reset.ax.set_facecolor(disabled_color)

            self.button_save.set_active(False)
            self.button_save.color = disabled_color
            self.button_save.ax.set_facecolor(disabled_color)

            self.button_load.set_active(False)
            self.button_load.color = disabled_color
            self.button_load.ax.set_facecolor(disabled_color)

        self.fig.canvas.draw_idle()

    def _save_button_callback(self):
        if not self.simulation_paused:
            return
        self.save_configs_to_file()

    def _load_button_callback(self):
        if not self.simulation_paused:
            return
        self.load_configs_from_file()

    def _on_pause_click(self, event):
        """Maneja el clic en el botón de Pausa."""
        if self.ani:
            self.ani.event_source.stop()

        self.simulation_paused = True
        self.current_speed_mode = 'pause'
        self._update_speed_buttons()

    def _on_play_click(self, event):
        """Maneja el clic en el botón de Play (velocidad normal)."""
        if not self.simulation_paused and self.current_speed_mode == 'play':
            return

        if self.ani:
            self.ani.event_source.stop()

        self.ani = self._create_animation(BASE_INTERVAL_MS)
        self.simulation_paused = False
        self.current_speed_mode = 'play'
        self._update_speed_buttons()

    def _on_turbo_click(self, event):
        """Maneja el clic en el botón de Turbo."""
        if not self.simulation_paused and self.current_speed_mode == 'turbo':
            return
        if self.ani:
            self.ani.event_source.stop()
        self.ani = self._create_animation(max(10, int(BASE_INTERVAL_MS / 10)))
        self.simulation_paused = False
        self.current_speed_mode = 'turbo'
        self._update_speed_buttons()

    def _on_reset_click(self, event):
        """Maneja el clic en el botón de Reset."""
        if not self.simulation_paused:
            return
        if self.ani:
            self.ani.event_source.stop()
        self.ani = self._create_animation(BASE_INTERVAL_MS)
        self.ani.event_source.stop()
        self.model = FireSimulationModel()
        self.im.set_array(self._build_display_image())
        self.stats_history = []
        self.line_empty.set_data([], [])
        self.line_grass.set_data([], [])
        self.line_burning.set_data([], [])
        self.line_burnt.set_data([], [])
        self.ax2.set_xlim(STATS_AXIS['xlim'])
        self._update_speed_buttons()
        self.fig.canvas.draw_idle()
        self.update_brush_text()

    def _reset_simulation_state(self):
        """Resetea la simulación a su estado inicial."""
        self.simulation_paused = True
        self.current_speed_mode = 'pause'
        if self.ani:
            self.ani.event_source.stop()
        self.model = FireSimulationModel()
        self.im.set_array(self._build_display_image())

        # Resetear estadísticas
        self.stats_history = []
        self.line_empty.set_data([], [])
        self.line_grass.set_data([], [])
        self.line_burning.set_data([], [])
        self.line_burnt.set_data([], [])
        self.ax2.set_xlim(STATS_AXIS['xlim'])

        # Actualizar estado de botones
        self._update_speed_buttons()
        self.fig.canvas.draw_idle()
        self.update_brush_text()

    def _update_animation(self, frame):
        """Actualiza la animación en cada frame."""
        if self.simulation_paused:
            return [self.im, self.line_empty, self.line_grass, self.line_burning, self.line_burnt]

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
            min_width = STATS_AXIS['xlim'][1]
            self.ax2.set_xlim(STATS_AXIS['xlim'][0], max(min_width, len(x_data)) + 1)

        return [self.im, self.line_empty, self.line_grass, self.line_burning, self.line_burnt]

    def _calculate_and_set_wind(self):
        """
        Convierte (Ángulo GUI, Velocidad GUI) a (Vector Normalizado, Intensidad 0-1) y actualiza el modelo.
        """
        angle_rad = np.deg2rad(self.wind_angle)
        norm_y = np.cos(angle_rad)
        norm_x = np.sin(angle_rad)
        self.model.wind_direction = [norm_y, -norm_x]
        max_speed = SLIDER_LIMITS["wind_speed"][1]
        intensity = self.wind_speed / max_speed if max_speed > 0 else 0
        self.model.wind_intensity = intensity

    def _update_wind_angle(self, val):
        """Se llama cuando el slider de ángulo cambia."""
        self.wind_angle = val
        self._calculate_and_set_wind()

    def _update_wind_speed(self, val):
        """Se llama cuando el slider de velocidad cambia."""
        self.wind_speed = val
        self._calculate_and_set_wind()

    def _update_humidity(self, val):
        self.model.humidity = val

    def show(self):
        """Muestra la interfaz gráfica."""
        plt.show()

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
        root.destroy()

    def load_configs_from_file(self):
        """Carga el estado del modelo desde un archivo NPZ."""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy compressed", "*.npz")],
            title="Cargar configuración de simulación"
        )
        if file_path:
            with np.load(file_path) as data:
                self.model.import_state(data)

                model_y, model_x = self.model.wind_direction
                model_intensity = self.model.wind_intensity
                self.wind_angle = np.rad2deg(np.arctan2(-model_x, model_y))
                if self.wind_angle < 0:
                    self.wind_angle += 360
                max_speed = SLIDER_LIMITS["wind_speed"][1]
                self.wind_speed = model_intensity * max_speed
                self._calculate_and_set_wind()

                self.slider_wind_angle.set_val(self.wind_angle)
                self.slider_wind_speed.set_val(self.wind_speed)
                self.slider_humidity.set_val(self.model.humidity)
                self.slider_temperature.set_val(self.model.temperature)
                self.slider_soil_moisture.set_val(self.model.soil_moisture)
                self.slider_grass_density.set_val(self.grass_density)

                self.simulation_paused = True
                self.current_speed_mode = 'pause'
                if self.ani:
                    self.ani.event_source.stop()
                self.ani = self._create_animation(BASE_INTERVAL_MS)
                self.ani.event_source.stop()
                self._update_speed_buttons()

                self.im.set_array(self._build_display_image())
                self.fig.canvas.draw()
        root.destroy()
