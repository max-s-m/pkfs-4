import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def logist_next(x_curr, r_val):
    return r_val * x_curr * (1 - x_curr)

def logist_map(x0, n_iter, r_val):
    plt_logist = np.zeros(n_iter + 1)
    plt_logist[0] = x0
    for i in range(n_iter):
        plt_logist[i + 1] = logist_next(plt_logist[i], r_val)
    return plt_logist


class ChaosApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1440x900")
        self.root.title("Модель хаосу")
        self.r = tk.DoubleVar(value=3.9)
        self.n_iter = tk.IntVar(value=100)
        self.x0_1 = tk.DoubleVar(value=0.2)
        self.d_x0 = tk.DoubleVar(value=1e-5)
        self.canv_plot_widget = None
        self.create_widgets()
        self.plot_logist()

    def create_widgets(self):
        self.bg_img = ImageTk.PhotoImage(Image.open("backgr.jpg").resize((1920, 1080)))
        self.bg_img_label = tk.Label(self.root, image=self.bg_img, bd=0)
        self.bg_img_label.place(x=0, y=0)

        self.frame_main = tk.Frame(self.root, bd=10, relief="raised")
        self.frame_main.pack(pady=(50, 0))

        self.title_label = tk.Label(self.frame_main, text="Модель хаосу 'Logistic Map'", font="Times 32")
        self.title_label.pack(pady=10)

        self.frame_controls = tk.Frame(self.frame_main)
        self.frame_controls.pack(pady=10, padx=30)

        self.frame_plot = tk.Frame(self.frame_main)
        self.frame_plot.pack(padx=(10), pady=(10, 30))

        self.formula_img = ImageTk.PhotoImage(Image.open("logistic_map.png").resize((300, 35)))
        self.formula_img_label = tk.Label(self.frame_controls, image=self.formula_img, bd=0)
        self.formula_img_label.grid(row=0, column=0, padx=(10, 10), pady=(5, 15))

        self.btn_quit = tk.Button(self.frame_controls, text="Вийти", font=('Times 15'), command=self.root.destroy)
        self.btn_quit.grid(row=0, column=1, padx=(0, 50), pady=(5, 15))

        tk.Label(self.frame_controls, text="Коефіцієнт приросту (r):", font=('Times 15')).grid(row=1, column=0, padx=(20, 0))
        tk.Scale(self.frame_controls, variable=self.r, from_=0.9, to=4.0, resolution=0.01, orient='horizontal', length=500, command=self.plot_refr).grid(row=2, column=0, padx=(20, 0), pady=(0, 30))

        tk.Label(self.frame_controls, text="К-ть ітерацій:", font=('Times 15')).grid(row=3, column=0, padx=(20, 0))
        tk.Scale(self.frame_controls, variable=self.n_iter, from_=10, to=200, orient='horizontal', length=500, command=self.plot_refr).grid(row=4, column=0, padx=(20, 0))

        tk.Label(self.frame_controls, text="Початкове значення (x0):", font=('Times 15')).grid(row=1, column=1, padx=(30, 10))
        tk.Scale(self.frame_controls, variable=self.x0_1, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', length=500, command=self.plot_refr).grid(row=2, column=1, padx=(30, 10), pady=(0, 30))

        tk.Label(self.frame_controls, text="Різниця початкових значень (d_x0):", font=('Times 15')).grid(row=3, column=1, padx=(30, 10))
        tk.Scale(self.frame_controls, variable=self.d_x0, from_=1e-9, to=2e-5, resolution=1e-9, orient='horizontal', length=500, command=self.plot_refr).grid(row=4, column=1, padx=(30, 10))

        self.fig_plot = Figure(figsize=(12, 4), dpi=100)
        self.chaos_plot = self.fig_plot.add_subplot(1, 1, 1)

        self.canv_plot = FigureCanvasTkAgg(self.fig_plot, master=self.frame_plot)
        self.canv_plot_widget = self.canv_plot.get_tk_widget()
        self.canv_plot_widget.pack(padx=10)

        self.chaos_plot.set_title("Графік моделі хаосу")
        self.chaos_plot.set_xlabel("")
        self.chaos_plot.set_ylabel("")
        self.chaos_plot.grid(True)
        self.fig_plot.tight_layout()
        self.canv_plot.draw()

    def plot_logist(self):
        r_val = self.r.get()
        n_iter_val = self.n_iter.get()
        x0_val = self.x0_1.get()
        dx_val = self.d_x0.get()
        x0_2 = x0_val + dx_val
        plt_logist_1 = logist_map(x0_val, n_iter_val, r_val)
        plt_logist_2 = logist_map(x0_2, n_iter_val, r_val)
        iter = np.arange(n_iter_val + 1)

        self.chaos_plot.clear()
        self.chaos_plot.plot(iter, plt_logist_1, 'b-', label=f'Графік 1 (x0={x0_val:.5f})', alpha=0.7)
        self.chaos_plot.plot(iter, plt_logist_2, 'r--', label=f'Графік 2 (x0={x0_2:.5f})', alpha=0.7)
        self.chaos_plot.set_title(f'Логістичне моделювання (r={r_val})')
        self.chaos_plot.set_xlabel('Ітерація')
        self.chaos_plot.set_ylabel('X')
        self.chaos_plot.legend()
        self.chaos_plot.grid(True)
        self.fig_plot.tight_layout()
        self.canv_plot.draw()

    def plot_refr(self, event=None):
        self.plot_logist()

    def run(self):
        self.root.mainloop()


def main():
    app = ChaosApp()
    app.run()


if __name__ == "__main__":
    main()