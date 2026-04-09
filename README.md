# Simulador Predictivo de Quemas Controladas en Pastizales

Este repositorio contiene el código fuente, diseño y documentación de un proyecto de tesis enfocado en el desarrollo de un modelo predictivo del avance de quemas controladas en pastizales y entornos rurales de Argentina en la región de la Pampa Húmeda. 
Las quemas controladas, ejecutadas bajo preceptos técnicos y científicos, son una herramienta fundamental en la ecología del fuego: permiten reducir la biomasa no deseada, fomentan la renovación de pastos de alta calidad forrajera y promueven la biodiversidad.
Este software se propone como una herramienta para ayudar al entendimiento y realización de quemas controladas al permitir a usuarios no técnicos la ejecución de un modelo predictivo que permita predecir el avance de una quema controlada bajo diferentes parámetros de clima configurables en tiempo real, como son la humedad, la dirección y velocidad del viento, la temperatura, etc.

---

## ⚙️ Parte 1: Diseño y Arquitectura del Modelo

El núcleo del modelo es un **Autómata Celulare (AC)**. Este paradigma computacional discreto destaca por su capacidad para simular dinámicas espacio-temporales complejas en paisajes heterogéneos con un bajo costo computacional, lo que lo hace un candidato perfecto para pruebas de campo o ejecución en sistemas de bajos recursos.

### Características del Autómata Celular
* **Topología Espacial:** El terreno se representa mediante una retícula bidimensional donde el estado de cada celda se define de manera discreta (ej. *no quemado*, *quemándose*, *quemado*).
* **Vecindarios:** El AC cuenta con un vecindario de Moore (8 celdas, sumando diagonales), permitiendo una propagación omnidireccional más realista del fuego.
* **Dinámica de Transición (Estocástica con Memoria):** El modelo es no determinista, por lo que diversas ejecuciones bajo las mismas condiciones de clima y terreno darán resultados diferentes. Las celdas mantienen contadores internos que registran el tiempo de combustión. La intensidad del fuego y la longitud de la llama de una celda modifican directamente la "capacidad de ignición" de las celdas adyacentes.

### Variables Físicas y Ambientales
La propagación del fuego en el modelo está regida por la interacción compleja de múltiples variables ambientales y topológicas implementadas en el motor de simulación:
* **Viento (Dirección e Intensidad):** Factor dominante del modelo. Funciona como un vector bidimensional que calcula la alineación geométrica del fuego, empujando e incrementando drásticamente la probabilidad de ignición en las celdas ubicadas a favor del viento.
* **Temperatura Ambiental:** Si la temperatura supera un umbral base, se añade un factor de penalización que incrementa la sensibilidad térmica y facilita la propagación del fuego a nuevas celdas.
* **Humedad Relativa / Ambiental:** Actúa como un factor global de amortiguación; a mayor humedad ambiental, el fuego tiene menor probabilidad de expandirse de una celda a otra.
* **Densidad de Vegetación (`grass_density`):** Representa la carga de biomasa disponible. Determina el tiempo exacto que una celda permanece en el estado "En combustión" antes de agotarse y pasar al estado "Quemado".
* **Sequedad Local del Combustible (`dryness_grid`):** Grilla espacial que permite definir un nivel de sequedad específico para el pasto en distintas zonas del mapa, funcionando como un multiplicador de propagación.
* **Cuerpos de Agua y Humedad del Suelo:** Las zonas marcadas como agua actúan como barreras incombustibles. Además, el modelo calcula un radio de efecto alrededor de estos cuerpos de agua, otorgando un "bonus" de humedad al suelo (`soil_moisture`) de las celdas cercanas, haciéndolas naturalmente más resistentes al inicio de nuevos focos.

---

## 💻 Parte 2: Implementación, Interfaz Gráfica y Validación

Para convertir el modelo matemático en una herramienta accesible, se desarrolló una interfaz interactiva que facilita la carga de variables y la visualización del ecosistema en tiempo real.

### Interfaz Gráfica de Usuario (GUI)
La interfaz permite a los usuarios interactuar con la retícula sin necesidad de conocimientos de programación.
* **Configuración de Escenarios:** Parametrización interactiva de variables meteorológicas (velocidad/dirección del viento, temperatura, humedad) y distribución topográfica con herramientas de "pincel" (para dibujar pasto, fuego, agua o niveles de sequedad).
* **Controles de Ejecución:** Funciones de reproducción, pausa, reinicio y control del reloj del modelo.
* **Importación/Exportación:** Capacidad de guardar el estado actual de la grilla y los parámetros para retomar simulaciones específicas.
* **Estadísticas:** Análisis en tiempo real del porcentaje del terreno afectado por el fuego.

### Verificación del Modelo
Se realizaron pruebas de software para garantizar que el modelo computacional hace exactamente lo que el diseño matemático dictamina:
* Control de contornos y límites espaciales de la matriz para evitar errores de desbordamiento de memoria.
* Comprobación de la lógica de estados estocástica.
* Validación de la inyección vectorial del viento en el Autómata Celular, garantizando un sesgo realista y físicamente correcto en el área quemada.

### Validación del Modelo
Se contrastaron los resultados del simulador con el comportamiento del fuego en el mundo real:
* **Morfología del fuego:** Comprobación de que las reglas estocásticas generen patrones de propagación irregulares, fractales y realistas, consistentes con la literatura de ecología del fuego.
* **Calibración empírica:** Ajuste de las probabilidades base de ignición apoyándose en datos de entornos rurales argentinos y pautas de comportamiento descritas por organismos oficiales (INTA).

---

## 📎 Créditos y Agradecimientos

* **Desarrollo del Modelo y Tesis:** Desarrollado como proyecto de tesis de grado/maestría por Denise Aeschbacher y Mariano Aguilar (Diseño del modelo) / Facundo Folis y Fausto Mansilla Sanz (Implementación).
* **Recursos Gráficos:** Los íconos y recursos visuales empleados en la interfaz gráfica interactiva fueron obtenidos de la plataforma **Flaticon** (www.flaticon.com).
