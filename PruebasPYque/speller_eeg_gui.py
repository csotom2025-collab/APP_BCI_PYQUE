"""
Speller EEG GUI (prototype) - Python + Tkinter
Características:
- Interfaz 5x5 con filas: vocales arriba, consonantes frecuentes, etc., y fila de controles (ESP, BORR, ENTER).
- Modo MANUAL: hacer clic en botones para "escribir".
- Modo SIMULADO P300: elige un objetivo (haz clic derecho sobre la letra para marcarlo como objetivo), pulsa "Start P300" y la app simula flashes y un clasificador que intenta adivinar la letra atendida.
- Puntos de integración con tu BCI real: sustituye la función `simulate_scores_with_eeg` por la referencia a tu clasificador (recibir vectores/score por estímulo).

Nota: prototipo pruebas de interfaz.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import random

# Configuración del teclado (prototipo)
# - Primera fila: vocales
# - Resto: consonantes ordenadas de más usadas a menos usadas en español (incluye Ñ)
GRID = [
    ["A", "E", "I", "O", "U"],
    ["S", "R", "N", "L", "D"],
    ["C", "T", "M", "P", "B"],
    ["G", "V", "Y", "Q", "H"],
    ["F", "Z", "J", "Ñ", "X"],
    ["K ", "W"," "]
]
GrindNumbers = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
    [" ", "0", " "],
]

CONTROLS = ["ESP", "BORR", "ENTER"]

FLASH_ROUNDS = 1      # cuántas rondas de flash por botón
FLASH_DURATION = 0.5 # segundos que dura cada flash
PAUSE_BETWEEN = 0.06  # pausa entre flashes

class SpellerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Speller EEG - Prototipo")

        self.target = None  # para simular la letra a la que el usuario "atiende"
        self.build_ui()
        self.lock = threading.Lock()
        self.running = False

    def build_ui(self):
        # Texto resultante en la parte superior y autoajustable
        self.output = tk.Text(self.root, height=4)
        self.output.pack(fill=tk.X, expand=True, side=tk.TOP, padx=10, pady=(10,0))

        # Frame contenedor horizontal
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.frame = tk.Frame(main_frame)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.buttons = {}

        # Configurar grid para autoescalado en panel de letras
        for r in range(len(GRID)):
            self.frame.rowconfigure(r, weight=1)
        for c in range(len(GRID[0])):
            self.frame.columnconfigure(c, weight=1)

        # Grid principal (letras)
        for r, row in enumerate(GRID):
            for c, label in enumerate(row):
                b = tk.Button(self.frame, text=label)
                b.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                b.config(width=6, height=3)
                b.config(font=("Arial", 14))
                b.config(wraplength=1)
                b.config(anchor="center")
                b.config(relief="raised")
                b.config(bg="SystemButtonFace")
                b.config(fg="black")
                b.config(activebackground="yellow")
                b.config(activeforeground="black")
                b.config(cursor="hand2")
                b.config(borderwidth=2)
                b.config(highlightthickness=0)
                b.config(takefocus=1)
                b.config(command=lambda L=label: self.on_click(L))
                b.bind('<Button-3>', lambda e, L=label: self.set_target(L))  # click derecho para marcar target
                self.buttons[label] = b

        # Panel numérico al lado derecho
        num_frame = tk.Frame(main_frame)
        num_frame.pack(side=tk.RIGHT, padx=(10,0), fill=tk.BOTH, expand=True)
        for r in range(len(GrindNumbers)):
            num_frame.rowconfigure(r, weight=1)
        for c in range(len(GrindNumbers[0])):
            num_frame.columnconfigure(c, weight=1)
        for r, row in enumerate(GrindNumbers):
            for c, label in enumerate(row):
                if label.strip() == "":
                    continue
                b = tk.Button(num_frame, text=label)
                b.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                b.config(width=6, height=3)
                b.config(font=("Arial", 14))
                b.config(wraplength=1)
                b.config(anchor="center")
                b.config(relief="raised")
                b.config(bg="SystemButtonFace")
                b.config(fg="black")
                b.config(activebackground="yellow")
                b.config(activeforeground="black")
                b.config(cursor="hand2")
                b.config(borderwidth=2)
                b.config(highlightthickness=0)
                b.config(takefocus=1)
                b.config(command=lambda L=label: self.on_click(L))
                b.bind('<Button-3>', lambda e, L=label: self.set_target(L))  # click derecho para marcar target
                self.buttons[label] = b

        # Controles escalables (solo pack en root, grid en el frame)
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, expand=True, padx=10, pady=(6,0))
        for i in range(len(CONTROLS)):
            ctrl_frame.columnconfigure(i, weight=1)
        for i, label in enumerate(CONTROLS):
            b = tk.Button(ctrl_frame, text=label)
            b.grid(row=0, column=i, padx=6, sticky="nsew")
            b.config(width=8, font=("Arial", 12), command=lambda L=label: self.on_control(L))
            self.buttons[label] = b


        # Botones de simulación / integración
        bottom = tk.Frame(self.root)
        bottom.pack(pady=8)

        self.start_btn = tk.Button(bottom, text="Start P300 (sim)", command=self.start_p300)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = tk.Button(bottom, text="Stop", command=self.stop_p300, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=6)

        self.info_lbl = tk.Label(bottom, text="Haz click derecho en una letra para marcarla como objetivo (sim)")
        self.info_lbl.grid(row=0, column=2, padx=12)

    def on_click(self, label):
        if label == 'ESP':
            self.output.insert(tk.END, ' ')
        elif label == 'BORR':
            content = self.output.get('1.0', tk.END)[:-1]
            self.output.delete('1.0', tk.END)
            self.output.insert('1.0', content[:-1])
        elif label == 'ENTER':
            self.output.insert(tk.END, '\n')
        else:
            self.output.insert(tk.END, label)

    def on_control(self, label):
        self.on_click(label)

    def set_target(self, label):
        # Click derecho: marca target para la simulación P300
        self.target = label
        self.info_lbl.config(text=f"Target simulado: {label}")

    def start_p300(self):
        if self.running:
            return
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        t = threading.Thread(target=self.p300_session)
        t.daemon = True
        t.start()

    def stop_p300(self):
        with self.lock:
            self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def p300_session(self):
        # Mensaje a escribir letra por letra
        mensaje = "HOLA MUNDO"
        self.escribir_siguiente_letra(mensaje, 0)
    
    def escribir_siguiente_letra(self, mensaje, indice):
        if indice >= len(mensaje) or not self.running:
            # Terminar cuando se hayan escrito todas las letras
            with self.lock:
                self.running = False
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            return
        letra = mensaje[indice]
        
        # Mostrar diálogo y continuar con la siguiente letra después
        messagebox.showinfo("Progreso", f"Penso en  '{letra if letra != ' ' else 'ESPACIO'}'.")

        if letra == " ":
            self.flash_button('ESP')
            time.sleep(FLASH_DURATION)
            self.root.after(0, lambda: self.on_click('ESP'))
        else:
            self.flash_button(letra)
            time.sleep(FLASH_DURATION)
            self.root.after(0, lambda: self.on_click(letra))

        
        # Programar la siguiente letra después de que se cierre el diálogo
        self.root.after(100, lambda: self.escribir_siguiente_letra(mensaje, indice + 1))

    def flash_button(self, label):
        # Cambia el color del botón brevemente para simular estímulo
        btn = self.buttons.get(label)
        if not btn:
            return
        # use after to avoid blocking mainloop
        def do_flash():
            orig = btn.cget('bg')
            btn.config(relief='sunken')
            btn.config(bg='yellow')
            self.root.update_idletasks()
            time.sleep(FLASH_DURATION)
            btn.config(relief='raised')
            btn.config(bg=orig)
            self.root.update_idletasks()  
        # run in main thread via after
        self.root.after(0, do_flash)

    def simulate_scores_with_eeg(self, flashed_label, true_target):
        """
        Simula cómo un clasificador EEG respondería al estímulo
        - Si el estímulo coincide con el target, devolvemos un score mayor en promedio
        - En un sistema real, aquí integrarías la lectura/feature-extraction/clasificador
        que devuelve un "score" o probabilidad por estímulo.
        """
        base_noise = random.gauss(0.0, 1.0)
        if true_target is None:
            # comportamiento sin target marcado: ruido
            return base_noise
        # si flashed_label == true_target: señal + ruido
        if flashed_label == true_target:
            return base_noise + random.uniform(2.0, 4.0)
        else:
            return base_noise + random.uniform(-0.5, 0.6)


if __name__ == '__main__':
    root = tk.Tk()
    app = SpellerGUI(root)
    root.mainloop()
