#from gui.src.discosscan import CLIGHT
from os import close, remove

import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib import colors
import numpy as np

from PyQt5.QtWidgets import QDialog, QVBoxLayout



#class GraphCreator(QtCore.QThread):
class GraphCreator:

    def __init__(self):

        # *** VERY IMPORTANT ***
        # A fig and ax should never created as static variables e.i. declared between the class name and the __init__ function
        # This because once the user quits from the application, the latter triggers a "segmentation fault" error which is associated to QFont of QTimer issues
        # This happens because the fig generated and attached to the canvas is never properly destroyed before closing the application
        # fig and ax must be created in the __init__ method or inside a function
        # to properly clear the memory before closing the application, fig and ax should be deleted by using the following comands:
        # <your_ax>.remove()
        # plt.close(<your_fig>)

        pass

   

    def create_MPL_empty_plot(self, title):

        # Regardless of the figure size, the figure then fills all the vertical layout if the latter has a big dimensions
        fig = plt.figure(figsize=(5, 3), dpi=100, facecolor=(241/255, 248/255, 255/255), layout='constrained')
        fig.suptitle('')
        ax = fig.add_subplot()
        # Set the limits for the x and y axes
        ax.set_xlim(0, 1024)
        ax.set_ylim(0, 10)
        # Optionally, set the title and labels
        ax.set_title(title)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Amplitude")
        # Set the background color of the plot area (axes) using RGB values
        ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)
        # Add a grid to the plot
        ax.grid(True)

        plt.tight_layout()

        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)

        return canvas

     

    def create_power_spectrum_plot(self, psd_pol0, psd_pol1, nchan, f0, bw, title):

        samplingFrequency = nchan
        samplingInterval = 1 / nchan
        freq = np.fft.fftfreq(samplingFrequency, samplingInterval)
        freq_abs_val = np.fft.fftshift((freq/1024*bw)+f0)

        # USIAMO Figure() DIRETTAMENTE INVECE DI plt.figure() PER EVITARE IL WARNING SULLE 20 FIGURE
        fig = Figure(figsize=(5, 3), dpi=100, facecolor=(241/255, 248/255, 255/255), layout='constrained')
        ax = fig.add_subplot(111)

        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Amplitude [AU]")
        ax.set_facecolor((241/255, 248/255, 255/255))

        line = [None] * 2
        lines = []

        (line[0], ) = ax.step(freq_abs_val, np.fft.fftshift(np.array(psd_pol0).real), where='mid', label='POL [LEFT]', linewidth=1)
        lines.append(line[0]) 
        (line[1], ) = ax.step(freq_abs_val, np.fft.fftshift(np.array(psd_pol1).real), where='mid', label='POL [RIGHT]', linewidth=1)
        lines.append(line[1]) 

        leg = ax.legend(fancybox=True, shadow=True)

        map_legend_to_ax = {}
        pickradius = 5

        for legend_line, ax_line in zip(leg.get_lines(), lines):
            legend_line.set_picker(pickradius)
            map_legend_to_ax[legend_line] = ax_line
        
        def on_pick(event):
            legend_line = event.artist
            if legend_line not in map_legend_to_ax:
                return
            ax_line = map_legend_to_ax[legend_line]
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        canvas = FigureCanvas(fig)
        canvas.mpl_connect('pick_event', on_pick)
        leg.set_draggable(True)

        toolbar = NavigationToolbar2QT(canvas, None)

        return canvas, toolbar



class AdvancedDiagnosticsDialog(QDialog):
    def __init__(self, parent=None, fft_raw=None, fft_clean=None, freqs=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced RFI & Pulsar QA Diagnostics")
        self.resize(1100, 750)

        layout = QVBoxLayout(self)

        # Figura Matplotlib 2x2
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 7))
        self.canvas = FigureCanvas(self.fig)
        
        # Toolbar di navigazione
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.fft_raw = fft_raw       # Matrice FFT o PSD 2D grezza
        self.fft_clean = fft_clean   # Matrice FFT o PSD 2D pulita
        self.freqs = freqs           # Array frequenze in MHz

        self.plot_all()

    

    def update_data_explicit(self, fft_raw, psd_clean_time_domain, freqs=None):
        """Riceve direttamente la matrice 2D temporale pulita ed esegue il re-plot."""
        self.fft_raw = fft_raw
        self.fft_clean = psd_clean_time_domain
        if freqs is not None:
            self.freqs = freqs
        
        self.plot_all()

    

    def update_data(self, fft_raw, fft_clean, freqs=None):
        """Aggiornamento in tempo reale."""
        self.fft_raw = fft_raw
        self.fft_clean = fft_clean
        if freqs is not None:
            self.freqs = freqs
        
        self.plot_all()

    

    def _extract_power_data(self, fft_matrix):
        """Estrae la matrice PSD 2D ordinata in frequenza (con FFT shift), Time Series 1D e Banda Passante 1D."""
        if fft_matrix is None:
            return None, None, None
        
        if not np.iscomplexobj(fft_matrix) and fft_matrix.ndim == 2:
            psd_2d = fft_matrix.copy()
        else:
            psd_2d = fft_matrix.real**2 + fft_matrix.imag**2

        # APPLICAZIONE FFTSHIFT SULL'ASSE DELLE FREQUENZE (axis=1)
        psd_2d_shifted = np.fft.fftshift(psd_2d, axes=1)
        bandpass_1d = np.mean(psd_2d_shifted, axis=0)
        time_series_1d = np.mean(psd_2d, axis=1)
        
        return psd_2d_shifted, bandpass_1d, time_series_1d
    


    def plot_all(self):
        # Pulizia assi
        for row in self.axes:
            for ax in row:
                ax.clear()

        # Estrazione metriche per i dati RAW e CLEAN
        psd_raw_2d, bp_raw, ts_raw = self._extract_power_data(self.fft_raw)
        psd_clean_2d, bp_clean, ts_clean = self._extract_power_data(self.fft_clean)

        # Se non ci sono dati caricati
        if psd_raw_2d is None:
            self.axes[0, 0].set_title("Banda Passante 1D (In attesa dati...)")
            self.axes[0, 1].set_title("Time Series 1D (In attesa dati...)")
            self.axes[1, 0].set_title("Waterfall RAW 2D (In attesa dati...)")
            self.axes[1, 1].set_title("Waterfall FILTERED 2D (In attesa dati...)")
            self.fig.tight_layout()
            self.canvas.draw()
            return

        n_time_raw, n_chan = psd_raw_2d.shape
        x_freq = self.freqs if self.freqs is not None else np.arange(n_chan)

        # --- 1. BANDA PASSANTE 1D (RAW vs FILTERED) ---
        self.axes[0, 0].plot(x_freq, bp_raw, color='orange', alpha=0.7, label='RAW', linewidth=1)
        if bp_clean is not None:
            self.axes[0, 0].plot(x_freq, bp_clean, color='#1f77b4', label='FILTERED', linewidth=1.2)
            
        self.axes[0, 0].set_title("Banda Passante Mediata (Frequency Spectrum)")
        self.axes[0, 0].set_xlabel("Frequenza [MHz]" if self.freqs is not None else "Canale")
        self.axes[0, 0].set_ylabel("Potenza Media")
        self.axes[0, 0].legend(loc='upper right', fontsize=8)
        self.axes[0, 0].grid(True, alpha=0.3)

        # --- 2. SERIE TEMPORALE INTEGRATA 1D (RAW vs FILTERED) ---
        time_axis_raw = np.arange(n_time_raw)
        self.axes[0, 1].plot(time_axis_raw, ts_raw, color='red', alpha=0.6, label='RAW', linewidth=0.8)
        
        if ts_clean is not None:
            n_time_clean = len(ts_clean)
            time_axis_clean = np.arange(n_time_clean)
            self.axes[0, 1].plot(time_axis_clean, ts_clean, color='navy', alpha=0.8, label='FILTERED', linewidth=0.8)
            
        self.axes[0, 1].set_title("Serie Temporale Integrata (Total Power vs Time)")
        self.axes[0, 1].set_xlabel(f"Iterazioni Temporali (Subset: {n_time_raw})")
        self.axes[0, 1].set_ylabel("Potenza Totale")
        self.axes[0, 1].legend(loc='upper right', fontsize=8)
        self.axes[0, 1].grid(True, alpha=0.3)

        # --- DOWNSAMPLING TEMPORALE PER VISUALIZZAZIONE ---
        max_display_rows = 600

        if n_time_raw > max_display_rows:
            block_raw = n_time_raw // max_display_rows
            raw_waterfall_ds = psd_raw_2d[:block_raw * max_display_rows].reshape(max_display_rows, block_raw, n_chan).mean(axis=1)
        else:
            raw_waterfall_ds = psd_raw_2d

        # --- 3. WATERFALL RAW 2D (CON VMIN/VMAX PROPRI) ---
        vmin_raw = np.percentile(raw_waterfall_ds, 2)
        vmax_raw = np.percentile(raw_waterfall_ds, 98)

        self.axes[1, 0].imshow(
            raw_waterfall_ds.T, aspect='auto', origin='lower', cmap='viridis',
            vmin=vmin_raw, vmax=vmax_raw, extent=[0, n_time_raw, x_freq[0], x_freq[-1]]
        )
        self.axes[1, 0].set_title("Waterfall Spettrogramma (RAW DATA)")
        self.axes[1, 0].set_xlabel(f"Iterazioni Temporali (0 - {n_time_raw})")
        self.axes[1, 0].set_ylabel("Frequenza [MHz]" if self.freqs is not None else "Canale")

        # --- 4. WATERFALL FILTERED 2D (CON VMIN/VMAX INDIPENDENTI SUI DATI PULITI) ---
        if psd_clean_2d is not None:
            n_time_clean = psd_clean_2d.shape[0]

            if n_time_clean > max_display_rows:
                block_clean = n_time_clean // max_display_rows
                clean_waterfall_ds = psd_clean_2d[:block_clean * max_display_rows].reshape(max_display_rows, block_clean, n_chan).mean(axis=1)
            else:
                clean_waterfall_ds = psd_clean_2d

            # VMIN/VMAX ricalcolati sui dati EFFETTIVAMENTE puliti
            vmin_clean = np.percentile(clean_waterfall_ds, 2)
            vmax_clean = np.percentile(clean_waterfall_ds, 98)

            self.axes[1, 1].imshow(
                clean_waterfall_ds.T, aspect='auto', origin='lower', cmap='viridis',
                vmin=vmin_clean, vmax=vmax_clean, extent=[0, n_time_clean, x_freq[0], x_freq[-1]]
            )
            self.axes[1, 1].set_title("Waterfall Spettrogramma (FILTERED DATA)")
            self.axes[1, 1].set_xlabel(f"Iterazioni Temporali (0 - {n_time_clean})")
        else:
            self.axes[1, 1].text(0.5, 0.5, "Premi FILTER per calcolare i dati puliti", 
                                 ha='center', va='center', transform=self.axes[1, 1].transAxes)
            self.axes[1, 1].set_title("Waterfall Spettrogramma (FILTERED - In Attesa)")
            self.axes[1, 1].set_xlabel("Iterazioni Temporali")

        self.axes[1, 1].set_ylabel("Frequenza [MHz]" if self.freqs is not None else "Canale")

        self.fig.tight_layout()
        self.canvas.draw()
