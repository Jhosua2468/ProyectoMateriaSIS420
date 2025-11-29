import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import random
import streamlit as st

GRID_SIZE = 30
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = 5
ALPHA = 0.3
GAMMA = 0.8
EPSILON_INITIAL = 0.3 
EPSILON_MIN = 0.05 
DECAY_RATE = 0.001  
PATIENCE = 35000 
MAX_STEPS = 500
EPISODIOS = 70001
print("Setup listo.")

class Figura:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.target = np.zeros((size, size))

    def generar_borde_circulo(self, centro=(15, 15), radio=8):
        self.target.fill(0)
        y, x = np.ogrid[:self.size, :self.size]
        mask = np.abs(np.sqrt((x - centro[1])**2 + (y - centro[0])**2) - radio) < 0.5 
        self.target[mask] = 1
        self.nombre = 'circulo_borde'
        return self.target

    def generar_borde_cuadrado(self, topleft=(10, 10), side=10):
        self.target.fill(0)
        self.target[topleft[0]:topleft[0]+side, topleft[1]] = 1
        self.target[topleft[0]:topleft[0]+side, topleft[1]+side-1] = 1
        self.target[topleft[0], topleft[1]:topleft[1]+side] = 1
        self.target[topleft[0]+side-1, topleft[1]:topleft[1]+side] = 1
        self.nombre = 'cuadrado_borde'
        return self.target

    def generar_borde_triangulo(self, apex=(5, 15), base_width=20, height=15):  
        self.target.fill(0)
        # Pico (topo)
        self.target[apex[0], apex[1]] = 1
        # Lado izquierdo (expande hacia abajo)
        for i in range(1, height):
            x_left = apex[1] - int((base_width // 2) * (i / height))
            self.target[apex[0] + i, x_left] = 1
        # Lado derecho (expande hacia abajo)
        for i in range(1, height):
            x_right = apex[1] + int((base_width // 2) * (i / height))
            self.target[apex[0] + i, x_right] = 1
        # Base horizontal abajo
        base_y = apex[0] + height - 1
        self.target[base_y, apex[1] - base_width//2 : apex[1] + base_width//2] = 1
        self.nombre = 'triangulo_borde'
        return self.target
    
    def generar_borde_rectangulo(self, topleft=(8, 8), width=11, height=20):
        self.target.fill(0)
        self.target[topleft[0]:topleft[0]+height, topleft[1]] = 1  # Izq
        self.target[topleft[0]:topleft[0]+height, topleft[1]+width-1] = 1  # Der
        self.target[topleft[0], topleft[1]:topleft[1]+width] = 1  # Arriba
        self.target[topleft[0]+height-1, topleft[1]:topleft[1]+width] = 1  # Abajo
        self.nombre = 'rectangulo_borde'
        return self.target

    def generar_borde_linea(self, start=(25, 5), end=(25, 25)):
        self.target.fill(0)
        y_start, x_start = start
        y_end, x_end = end
        if y_start == y_end:  # Horizontal
            self.target[y_start, x_start:x_end] = 1
        else:  # Vertical
            self.target[y_start:y_end, x_start] = 1
        self.nombre = 'linea_borde'
        return self.target

    def similitud(self, canvas):
        inter = np.sum(self.target * (canvas > 0))
        union = np.sum(self.target) + np.sum(canvas > 0) - inter
        return (inter / union * 100) if union > 0 else 0

    def render(self):
        plt.imshow(self.target, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Bordes: {self.nombre}')
        plt.show()

fig = Figura()
fig.generar_borde_circulo()  # cambio manual
fig.render()
print(f"Figura: {fig.nombre}")

class Lienzo:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.canvas = np.zeros((size, size))
        self.pos = [size // 2, size // 2]
        self.target = None
        self.target_pos = (0, 0)
        self.step_count = 0

    def reset(self):
        self.canvas = np.zeros((self.size, self.size))
        self.pos = [self.size // 2, self.size // 2]
        self.step_count = 0
        return self._get_state()

    def step(self, action, target=None):
        self.step_count += 1
        reward = -0.05

        y, x = self.pos
        in_zone = (self.target_pos[0] <= y < self.target_pos[0] + 30 and self.target_pos[1] <= x < self.target_pos[1] + 30)

        if action == 4:
            if self.canvas[y, x] == 0:
                self.canvas[y, x] = 1
                reward += 0.5  # Dibuja nuevo punto
                if in_zone:
                    reward += 3.5  # si el dibujo está en zona objetivo
                if target is not None and target.target[y, x] == 1:
                    reward += 5.0  # Bono por dibujar en lugar correcto
                else:
                    reward -= 25.5  # Más penaliza ruido
            else:
                reward -= 4.5  # Penaliza por dibujar en punto ya dibujado
        else:
            
            dy, dx = ACTIONS[action]
            self.pos[0] = np.clip(self.pos[0] + dy, 0, self.size - 1) 
            self.pos[1] = np.clip(self.pos[1] + dx, 0, self.size - 1)
            reward -= 0.05 # Costo por moverse
            if in_zone:
                reward += 0.09  # Pequeño bono por moverse en zona objetivo

        state = self._get_state()
        done = self.step_count >= MAX_STEPS
        if target is not None:
            sim = target.similitud(self.canvas)
            reward += sim / 20  # bono proporcional a la similitud
            if sim > 75:
                reward += 80  # Bono grande por alta similitud
                done = True
        return state, reward, done

    def _get_state(self):
        sum_drawn = int(np.sum(self.canvas))
        return (int(self.pos[0]), int(self.pos[1]), sum_drawn)

    def render_comparison(self, target):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(target.target, cmap='gray')
        axs[0].set_title('Target Borde')
        axs[1].imshow(self.canvas, cmap='gray')
        axs[1].set_title(f'Agente (Pasos: {self.step_count})')
        plt.show()

class Agente:
    def __init__(self, n_actions=N_ACTIONS):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions

    def elegir_accion(self, state, epsilon=EPSILON_INITIAL):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        # Ecuación de bellman Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
        best_next = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + GAMMA * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += ALPHA * td_error

    def guardar(self, nombre):
        with open(f'q_{nombre}.pkl', 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Guardado normal: q_{nombre}.pkl")

    def guardar_best(self, best_q_dict, nombre):  
        with open(f'q_best_{nombre}.pkl', 'wb') as f:
            pickle.dump(best_q_dict, f)
        print(f"Guardado BEST: q_best_{nombre}.pkl (mejor precisión)")

    def cargar(self, nombre):
        try:
            with open(f'q_{nombre}.pkl', 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
                loaded = pickle.load(f)
                for k, v in loaded.items():
                    self.q_table[k] = v
            print(f"Cargado normal: q_{nombre}.pkl")
        except FileNotFoundError:
            print("No modelo normal; entrena.")

    def cargar_best(self, nombre):  
        try:
            with open(f'q_best_{nombre}.pkl', 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
                loaded = pickle.load(f)
                for k, v in loaded.items():
                    self.q_table[k] = v
            print(f"Cargado BEST: q_best_{nombre}.pkl")
        except FileNotFoundError:
            print("No best modelo; usa cargar normal.")


def componer_casa():
    try:
        agente_cuad = Agente()
        agente_cuad.cargar_best('cuadrado_borde')  # Usa best Q
        agente_tri = Agente()
        agente_tri.cargar_best('triangulo_borde')  # Usa best Q
    except FileNotFoundError:
        print("Error: Entrena best para cuadrado/triángulo primero.")
        return np.zeros((60, 60))

    big_size = 60
    big_canvas = np.zeros((big_size, big_size))

    # Cuadrado (pared abajo)
    fig_cuad = Figura()
    fig_cuad.generar_borde_cuadrado()
    lienzo_cuad = Lienzo(GRID_SIZE)
    state = lienzo_cuad.reset()
    done = False
    for _ in range(MAX_STEPS):
        action = agente_cuad.elegir_accion(state)
        next_state, _, done = lienzo_cuad.step(action, fig_cuad)
        state = next_state
        if done: break
    offset_cuad = (25, 15)  # Abajo centro
    big_canvas[offset_cuad[0]:offset_cuad[0]+GRID_SIZE, offset_cuad[1]:offset_cuad[1]+GRID_SIZE] = lienzo_cuad.canvas

    # Triángulo (techo arriba)
    fig_tri = Figura()
    fig_tri.generar_borde_triangulo()
    lienzo_tri = Lienzo(GRID_SIZE)
    state = lienzo_tri.reset()
    done = False
    for _ in range(MAX_STEPS):
        action = agente_tri.elegir_accion(state)
        next_state, _, done = lienzo_tri.step(action, fig_tri)
        state = next_state
        if done: break
    offset_tri = (10, 15)  # Arriba centro
    big_canvas[offset_tri[0]:offset_tri[0]+GRID_SIZE, offset_tri[1]:offset_tri[1]+GRID_SIZE] += lienzo_tri.canvas * 0.5

    return big_canvas
def componer_arbol():
    try:
        agente_rect = Agente()
        agente_rect.cargar_best('rectangulo_borde')  # Tronco
        agente_circ = Agente()
        agente_circ.cargar_best('circulo_borde')  # Copa
    except FileNotFoundError:
        print("Error: Entrena best para rectangulo/circulo primero.")
        return np.zeros((60, 60))

    big_size = 60
    big_canvas = np.zeros((big_size, big_size))

    # Rectángulo (tronco abajo)
    fig_rect = Figura()
    fig_rect.generar_borde_rectangulo((10, 10), width=6, height=15)  # Delgado alto
    lienzo_rect = Lienzo(GRID_SIZE)
    state = lienzo_rect.reset()
    done = False
    for _ in range(MAX_STEPS):
        action = agente_rect.elegir_accion(state)
        next_state, _, done = lienzo_rect.step(action, fig_rect)
        state = next_state
        if done: break
    offset_rect = (30, 25)  # Abajo centro
    big_canvas[offset_rect[0]:offset_rect[0]+GRID_SIZE, offset_rect[1]:offset_rect[1]+GRID_SIZE] = lienzo_rect.canvas

    # Círculo (copa arriba)
    fig_circ = Figura()
    fig_circ.generar_borde_circulo((15, 15), radio=10)  # Grande
    lienzo_circ = Lienzo(GRID_SIZE)
    state = lienzo_circ.reset()
    done = False
    for _ in range(MAX_STEPS):
        action = agente_circ.elegir_accion(state)
        next_state, _, done = lienzo_circ.step(action, fig_circ)
        state = next_state
        if done: break
    offset_circ = (10, 20)  # Arriba del tronco
    big_canvas[offset_circ[0]:offset_circ[0]+GRID_SIZE, offset_circ[1]:offset_circ[1]+GRID_SIZE] += lienzo_circ.canvas * 0.5

    return big_canvas


# Lista de composiciones (de Celda 7)
COMPOSITES = {
    'casa': componer_casa,
    'arbol': componer_arbol
    # Agrega más: 'auto': componer_auto, etc.
}

st.title("🖌️ App Dibujante: Figuras Compuestas")
st.write("Selecciona composiciones para generar en lienzo grande.")

# Formulario: Multiselect de composiciones
seleccionadas = st.multiselect(
    "Elige figuras compuestas (e.g., casa, arbol):",
    options=list(COMPOSITES.keys()),
    default=[]  # Vacío inicial
)

if st.button("Generar Dibujo"):
    if seleccionadas:
        big_size = 120  # Lienzo extra grande para múltiples
        big_canvas = np.zeros((big_size, big_size))
        progress_bar = st.progress(0)
        status = st.empty()

        for i, nom in enumerate(seleccionadas):
            status.text(f"Generando {nom}... ({i+1}/{len(seleccionadas)})")
            canvas = COMPOSITES[nom]()  # Llama función
            if canvas is not None:
                # Offset auto (grid simple para no superponer)
                row, col = divmod(i, 2)  # 2x2 grid
                off_y = row * 40
                off_x = col * 40
                big_canvas[off_y:off_y+60, off_x:off_x+60] += canvas * 0.7  # Overlay con fade
            progress_bar.progress((i + 1) / len(seleccionadas))

        status.text("¡Dibujo listo!")
        progress_bar.progress(1.0)

        # Muestra imagen generada
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(big_canvas, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Dibujo Compuesto: {', '.join(seleccionadas)}")
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Selecciona al menos una composición.")

# Info
st.info("Entrena figuras primero (notebook). Expande COMPOSITES para más.")