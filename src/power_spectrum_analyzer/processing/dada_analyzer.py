import numpy as np
import time
import concurrent.futures
import os

#from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import pyqtSignal, QThread

from scipy.signal import medfilt


import os
import concurrent.futures



class CreateRawSpectrum(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    result_signal_psd = pyqtSignal(object, object, object, object)

    def __init__(self, iterations, npoints, nchan, data, parent=None):
        super().__init__(parent)

        self.iterations = iterations
        self.npoints = npoints
        self.nchan = nchan
        self.data = data
        
    def run (self):
        t_start_total = time.perf_counter()

        self.update_statusbar_signal.emit('FFT calculation started...')   
        self.update_button_signal.emit(False)     

        rows_per_step = self.npoints  # 1024
        iterations = self.iterations

        # 1. Reshape e centraggio a int16 (in-place dove possibile)
        t0 = time.perf_counter()
        data = self.data[:iterations * rows_per_step].reshape(iterations, rows_per_step, 4)
        
        # Estraiamo direttamente i canali I/Q centrandoli su 128 in formato float32 per i complessi
        i0 = data[:, :, 0].astype(np.float32) - 128.0
        q0 = data[:, :, 1].astype(np.float32) - 128.0
        i1 = data[:, :, 2].astype(np.float32) - 128.0
        q1 = data[:, :, 3].astype(np.float32) - 128.0
        t_prep = time.perf_counter() - t0

        # 2. Costruzione veloce del segnale complesso (view diretta)
        t0 = time.perf_counter()
        complex_pol0 = i0 + 1j * q0
        complex_pol1 = i1 + 1j * q1
        t_complex = time.perf_counter() - t0

        # 3. Calcolo FFT su asse frequenza (axis=1)
        t0 = time.perf_counter()
        fft_array_pol0 = np.fft.fft(complex_pol0, axis=1)
        fft_array_pol1 = np.fft.fft(complex_pol1, axis=1)
        t_fft = time.perf_counter() - t0

        # 4. Spettro di potenza (PSD) e Media integrata
        # Nota: (real^2 + imag^2) evita la sqrt interna di np.abs()
        t0 = time.perf_counter()
        self.update_statusbar_signal.emit('FFT calculation completed. Squaring coefficients...')
        self.update_progressbar_range.emit(0, 0)

        pol0_psd_tot = np.mean(fft_array_pol0.real**2 + fft_array_pol0.imag**2, axis=0)
        pol1_psd_tot = np.mean(fft_array_pol1.real**2 + fft_array_pol1.imag**2, axis=0)
        t_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE SPETTRO GREZZO (CreateRawSpectrum) ===")
        print(f" Iterazioni elaborate:              {iterations}")
        print(f" 1. Prepartazione Dati e Centraggio:{t_prep:.3f} s")
        print(f" 2. Costruzione Segnale Complesso:  {t_complex*1000:.2f} ms")
        print(f" 3. Calcolo FFT (Pol0 + Pol1):      {t_fft:.3f} s")
        print(f" 4. Calcolo PSD e Integrazione Media:{t_psd*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE SPETTRO GREZZO:       {t_total:.3f} s")
        print("==========================================================\n")

        # Store results
        self.fft_array_pol0 = fft_array_pol0
        self.fft_array_pol1 = fft_array_pol1
        self.pol0_psd_tot = pol0_psd_tot
        self.pol1_psd_tot = pol1_psd_tot

        
        # Imposta temporaneamente la stampa completa di NumPy
        with np.printoptions(threshold=np.inf, precision=4, suppress=True):
            print("\n=== PSD GREZZA POL0 (1024 canali) ===")
            print(self.pol0_psd_tot[190:195])
    
            print("\n=== PSD GREZZA POL1 (1024 canali) ===")
            print(self.pol1_psd_tot[190:195])




        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(self.iterations)

        self.result_signal_psd.emit(self.pol0_psd_tot, self.pol1_psd_tot, fft_array_pol0, fft_array_pol1)

        self.update_button_signal.emit(True)  
        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)



class CreateFilteredSpectrum(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    result_signal_psd = pyqtSignal(object, object, object, object)

    def __init__(self, fft_pol0, fft_pol1, channels_pol0, channels_pol1, outliers_indices_pol0, outliers_indices_pol1, parent=None):
        super().__init__(parent)

        self.fft_pol0 = fft_pol0
        self.fft_pol1 = fft_pol1
        self.channels_pol0 = channels_pol0
        self.channels_pol1 = channels_pol1
        self.outliers_indices_pol0 = outliers_indices_pol0
        self.outliers_indices_pol1 = outliers_indices_pol1

    def _find_safe_channels_fast(self, outliers_indices, nchan=1024, min_dist=5):
        if not outliers_indices:
            return []

        is_safe = np.ones(nchan, dtype=bool)
        for out in outliers_indices:
            start = max(0, out - min_dist + 1)
            end = min(nchan, out + min_dist)
            is_safe[start:end] = False

        safe_channels = []
        all_indices = np.arange(nchan)

        for out_idx in outliers_indices:
            distances = np.abs(all_indices - out_idx)
            distances[~is_safe] = 999999
            best_idx = np.argmin(distances)
            
            if distances[best_idx] >= 999999:
                best_idx = (out_idx + 10) % nchan
                
            safe_channels.append(int(best_idx))

        return safe_channels

    def _repair_polarization_vectorized(self, fft_data, outliers_indices, safe_channels):
        """
        Riparazione vettoriale istantanea senza cicli for explicit per l'estrazione casuale.
        """
        if not outliers_indices:
            return fft_data.copy()

        num_iterations = fft_data.shape[0]
        num_outliers = len(outliers_indices)

        # 1. Estrazione in un solo colpo dei moduli dei canali sani
        # safe_mags ha shape: (num_iterations, num_outliers)
        safe_mags = np.abs(fft_data[:, safe_channels])

        # 2. Calcolo vettoriale di medie e deviazioni standard lungo l'asse del tempo (axis=0)
        m = np.mean(safe_mags, axis=0)  # shape: (num_outliers,)
        s = np.std(safe_mags, axis=0)   # shape: (num_outliers,)

        # 3. Generazione MATRICIALE della distribuzione Gaussiana Standard N(0, 1)
        # Una singola chiamata in C per l'intero blocco!
        z_random = np.random.standard_normal((num_iterations, num_outliers))

        # 4. Trasformazione in N(m, s) e clamp a valore positivo safe
        new_mags = np.maximum(m + z_random * s, 1e-9)

        # 5. Modulo originale dei canali corrotti
        orig_mags = np.abs(fft_data[:, outliers_indices]) + 1e-9

        # 6. Sostituzione in-place matriciale dei coefficienti complessi
        clean_fft = fft_data.copy()
        clean_fft[:, outliers_indices] *= (new_mags / orig_mags)

        return clean_fft

    def run(self):
        t_start_total = time.perf_counter()

        self.update_statusbar_signal.emit('Reconstructing the FFT coefficients...')   
        self.update_button_signal.emit(False)  
        self.update_progressbar_range.emit(0, 0)

        # 1. Ricerca canali sani (distanti almeno 5 canali)
        t0 = time.perf_counter()
        self.channels_pol0 = self._find_safe_channels_fast(self.outliers_indices_pol0, min_dist=5)
        self.channels_pol1 = self._find_safe_channels_fast(self.outliers_indices_pol1, min_dist=5)
        t_safe_channels = time.perf_counter() - t0

        num_iterations = self.fft_pol0.shape[0]

        # 2. Riparazione Vettoriale Gaussiana Matrice Unica
        t0 = time.perf_counter()
        clean_fft_pol0 = self._repair_polarization_vectorized(self.fft_pol0, self.outliers_indices_pol0, self.channels_pol0)
        clean_fft_pol1 = self._repair_polarization_vectorized(self.fft_pol1, self.outliers_indices_pol1, self.channels_pol1)
        t_repair = time.perf_counter() - t0

        # 3. Calcolo PSD finale vettoriale
        t0 = time.perf_counter()
        self.update_statusbar_signal.emit('Squaring and integrating...')
        
        psd0_tot = np.mean(clean_fft_pol0.real**2 + clean_fft_pol0.imag**2, axis=0)
        psd1_tot = np.mean(clean_fft_pol1.real**2 + clean_fft_pol1.imag**2, axis=0)
        t_reconstruct_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE PULIZIA GAUSSIANA VETTORIALE (CreateFilteredSpectrum) ===")
        print(f" Iterazioni elaborate:                {num_iterations}")
        print(f" 1. Ricerca Canali Sani (min dist 5): {t_safe_channels*1000:.2f} ms")
        print(f" 2. Replacement Gaussiano Matriciale: {t_repair*1000:.2f} ms")
        print(f" 3. Calcolo PSD Finale:                {t_reconstruct_psd*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE PULIZIA:                {t_total:.3f} s")
        print("==========================================================\n")

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(100)   

        self.result_signal_psd.emit(psd0_tot, psd1_tot, clean_fft_pol0, clean_fft_pol1)

        self.update_button_signal.emit(True)  
        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)

        # Imposta temporaneamente la stampa completa di NumPy
        with np.printoptions(threshold=np.inf, precision=4, suppress=True):
            print("\n=== PSD RIPARATA POL0 (1024 canali) ===")
            print(psd0_tot[190:195])
    
            print("\n=== PSD RIPARATA POL1 (1024 canali) ===")
            print(psd1_tot[190:195])





class CreateFilteredSpectrum_old(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    result_signal_psd = pyqtSignal(object, object, object, object)

    def __init__(self, fft_pol0, fft_pol1, channels_pol0, channels_pol1, outliers_indices_pol0, outliers_indices_pol1, parent=None):
        super().__init__(parent)

        self.fft_pol0 = fft_pol0
        self.fft_pol1 = fft_pol1
        self.channels_pol0 = channels_pol0
        self.channels_pol1 = channels_pol1
        self.outliers_indices_pol0 = outliers_indices_pol0
        self.outliers_indices_pol1 = outliers_indices_pol1

    def _find_safe_channels_fast(self, outliers_indices, nchan=1024, min_dist=10):
        if not outliers_indices:
            return []

        is_safe = np.ones(nchan, dtype=bool)
        for out in outliers_indices:
            start = max(0, out - min_dist + 1)
            end = min(nchan, out + min_dist)
            is_safe[start:end] = False

        safe_channels = []
        all_indices = np.arange(nchan)

        for out_idx in outliers_indices:
            distances = np.abs(all_indices - out_idx)
            distances[~is_safe] = 999999
            best_idx = np.argmin(distances)
            
            if distances[best_idx] >= 999999:
                best_idx = (out_idx + 10) % nchan
                
            safe_channels.append(int(best_idx))

        return safe_channels

    def run(self):
        t_start_total = time.perf_counter()

        self.update_statusbar_signal.emit('Reconstructing the FFT coefficients...')   
        self.update_button_signal.emit(False)  
        self.update_progressbar_range.emit(0, 0)

        # 1. Ricerca rapida dei canali sani
        t0 = time.perf_counter()
        self.channels_pol0 = self._find_safe_channels_fast(self.outliers_indices_pol0)
        self.channels_pol1 = self._find_safe_channels_fast(self.outliers_indices_pol1)
        t_safe_channels = time.perf_counter() - t0

        # 2. Copia dei coefficienti FFT originali (lavoriamo direttamente sui complessi)
        t0 = time.perf_counter()
        clean_fft_pol0 = self.fft_pol0.copy()
        clean_fft_pol1 = self.fft_pol1.copy()
        t_copy = time.perf_counter() - t0

        num_iterations = self.fft_pol0.shape[0]

        # 3. Riparazione Pol0 e Pol1 in-place (Senza calcolare le fasi!)
        t0 = time.perf_counter()
        for out_idx, safe_idx in zip(self.outliers_indices_pol0, self.channels_pol0):
            # Magnitudo canale sano
            safe_mag = np.abs(self.fft_pol0[:, safe_idx])
            m, s = np.mean(safe_mag), np.std(safe_mag)
            
            # Generazione nuova magnitudo
            new_mag = np.maximum(np.random.normal(m, s, size=num_iterations), 1e-9)
            
            # Magnitudo originale del canale corrotto
            orig_mag = np.abs(self.fft_pol0[:, out_idx]) + 1e-9
            
            # Riscalatura diretta dei numeri complessi (mantiene la fase senza usare np.angle/np.exp)
            clean_fft_pol0[:, out_idx] = self.fft_pol0[:, out_idx] * (new_mag / orig_mag)

        for out_idx, safe_idx in zip(self.outliers_indices_pol1, self.channels_pol1):
            safe_mag = np.abs(self.fft_pol1[:, safe_idx])
            m, s = np.mean(safe_mag), np.std(safe_mag)
            
            new_mag = np.maximum(np.random.normal(m, s, size=num_iterations), 1e-9)
            orig_mag = np.abs(self.fft_pol1[:, out_idx]) + 1e-9
            
            clean_fft_pol1[:, out_idx] = self.fft_pol1[:, out_idx] * (new_mag / orig_mag)
            
        t_repair = time.perf_counter() - t0

        # 4. Calcolo PSD finale direttamente dai complessi puliti
        t0 = time.perf_counter()
        self.update_statusbar_signal.emit('Squaring and integrating...')
        
        psd0_tot = np.mean(np.abs(clean_fft_pol0)**2, axis=0)
        psd1_tot = np.mean(np.abs(clean_fft_pol1)**2, axis=0)
        t_reconstruct_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE PULIZIA OPT2 (CreateFilteredSpectrum) ===")
        print(f" Iterazioni elaborate:              {num_iterations}")
        print(f" 1. Ricerca Canali Sani:            {t_safe_channels*1000:.2f} ms")
        print(f" 2. Copia Buffer FFT:               {t_copy*1000:.2f} ms")
        print(f" 3. Replacement In-Place (NO Trigonometria): {t_repair*1000:.2f} ms")
        print(f" 4. Calcolo PSD Finale:             {t_reconstruct_psd:.3f} s")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE PULIZIA:              {t_total:.3f} s")
        print("==========================================================\n")

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(100)   

        self.result_signal_psd.emit(psd0_tot, psd1_tot, clean_fft_pol0, clean_fft_pol1)

        self.update_button_signal.emit(True)  
        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)




     

class DetectOutliers(QThread):

    # Signal to notify the main thread to disable/enable the button in the interface
    update_button_signal = pyqtSignal(bool)
    # Signal to notify the main thread (UI) to update the status bar in the interface 
    update_statusbar_signal = pyqtSignal(str)
    # Signal to notify the main thread (UI) to update the progressbar current value
    update_progressbar_value = pyqtSignal(int)
     # Signal to notify the main thread (UI) to update the progressbar current range
    update_progressbar_range = pyqtSignal(int, int)
    # Signal to notify the main thread with the result of the processing
    outliers_signal_psd = pyqtSignal(object, object)

    def __init__(self, power_spectrum_pol0, chunk_size_pol0, threshold_pol0,
                 power_spectrum_pol1, chunk_size_pol1, threshold_pol1, parent=None):
        super().__init__(parent)

        self.power_spectrum_pol0 = np.array(power_spectrum_pol0)
        self.chunk_size_pol0 = chunk_size_pol0
        self.threshold_pol0 = threshold_pol0

        self.power_spectrum_pol1 = np.array(power_spectrum_pol1)
        self.chunk_size_pol1 = chunk_size_pol1
        self.threshold_pol1 = threshold_pol1

    def _find_outliers(self, spectrum, chunk_size, threshold):
        n = len(spectrum)
        # Calcoliamo quanti elementi avanzano se n non � divisibile per chunk_size
        padding_size = (chunk_size - (n % chunk_size)) % chunk_size
        
        # Pad con NaN per non alterare le statistiche e mantenere la forma rettangolare
        padded_spectrum = np.append(spectrum, np.full(padding_size, np.nan))
        chunks = padded_spectrum.reshape(-1, chunk_size)

        # 1. Calcolo Mediana e MAD per ogni riga (chunk)
        # Usiamo nanmedian per ignorare il padding finale
        medians = np.nanmedian(chunks, axis=1, keepdims=True)
        mads = np.nanmedian(np.abs(chunks - medians), axis=1, keepdims=True)
        
        # 2. Modified Z-Score (lo 0.6745 serve per allinearsi alla std gaussiana)
        # Aggiungiamo un epsilon 1e-9 per evitare divisioni per zero
        mod_z_scores = 0.6745 * (chunks - medians) / (mads + 1e-9)
        
        # 3. Creazione maschera boolean
        mask = np.abs(mod_z_scores) > threshold
        
        # 4. Ricostruiamo gli indici originali
        outlier_indices = np.where(mask.flatten()[:n])[0]
        
        return outlier_indices.tolist()

    def run(self):

        self.update_statusbar_signal.emit('Inspecting the Power Spectrum for outliers...')
        self.update_button_signal.emit(False)
        self.update_progressbar_range.emit(0, 0)

        # Detect outliers for both polarizations
        outlier_indices_pol0 = self._find_outliers(
            self.power_spectrum_pol0,
            self.chunk_size_pol0,
            self.threshold_pol0
        )

        outlier_indices_pol1 = self._find_outliers(
            self.power_spectrum_pol1,
            self.chunk_size_pol1,
            self.threshold_pol1
        )

        print("Outliers pol0:", outlier_indices_pol0)
        print("Outliers pol1:", outlier_indices_pol1)

        # Emit results to main thread
        self.outliers_signal_psd.emit(outlier_indices_pol0, outlier_indices_pol1)

        self.update_statusbar_signal.emit('Outlier detection completed.')
        self.update_button_signal.emit(True)
        self.update_progressbar_value.emit(100)



class CreateFilteredTimeDomainSignal_(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    # Emissione dei 4 buffer temporali (I0, Q0, I1, Q1) e del fattore di scala calcolato
    result_signal_filtered_time_domain = pyqtSignal(object, object, object, object, float)

    def __init__(self, fft_corrected_pol0, fft_corrected_pol1, outliers_index_pol0, outliers_index_pol1, parent=None):
        super().__init__(parent)
        self.fft_corrected_pol0 = fft_corrected_pol0
        self.fft_corrected_pol1 = fft_corrected_pol1
        self.outliers_index_pol0 = outliers_index_pol0
        self.outliers_index_pol1 = outliers_index_pol1

    def safe_to_uint8(self, ifft0, ifft1, margin=127.0):
        """
        Calcola il fattore moltiplicativo massimo teorico per utilizzare 
        il 100% della dinamica a 8-bit senza produrre clipping.
        """
        # 1. Trova il picco massimo assoluto tra parti Reali e Immaginarie di entrambe le polarizzazioni
        max_val_0 = max(np.max(np.abs(ifft0.real)), np.max(np.abs(ifft0.imag)))
        max_val_1 = max(np.max(np.abs(ifft1.real)), np.max(np.abs(ifft1.imag)))
        global_max = max(max_val_0, max_val_1)
        
        # 2. Calcola il fattore di scala ottimale (spinto al massimo prima del clip)
        if global_max > 0:
            optimal_scale = margin / global_max
        else:
            optimal_scale = 1.0

        # 3. Trasformazione lineare e quantizzazione uint8 (corretta indentazione)
        r_uint8_0 = np.clip(ifft0.real * optimal_scale + 128.5, 0, 255).astype(np.uint8)
        i_uint8_0 = np.clip(ifft0.imag * optimal_scale + 128.5, 0, 255).astype(np.uint8)
        
        r_uint8_1 = np.clip(ifft1.real * optimal_scale + 128.5, 0, 255).astype(np.uint8)
        i_uint8_1 = np.clip(ifft1.imag * optimal_scale + 128.5, 0, 255).astype(np.uint8)
            
        return r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1, optimal_scale

    def run(self):
        t_start_total = time.perf_counter()

        self.update_button_signal.emit(False)
        self.update_statusbar_signal.emit('Ricostruzione del segnale tempo-variante (8-bit uint)...')

        # 1. Ripristino del centraggio dello spettro ed IFFT vettoriale su asse 1
        t0 = time.perf_counter()
        
        # De-centraggio spettrale necessario se la FFT originale è stata processata con fftshift
        fft_prep_pol0 = np.fft.ifftshift(self.fft_corrected_pol0, axes=1)
        fft_prep_pol1 = np.fft.ifftshift(self.fft_corrected_pol1, axes=1)

        ifft0 = np.fft.ifft(fft_prep_pol0, axis=1)
        ifft1 = np.fft.ifft(fft_prep_pol1, axis=1)
        t_ifft = time.perf_counter() - t0

        # 2. Conversione sicura a uint8 con scala dinamica al massimo teorico
        t0 = time.perf_counter()
        r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1, optimal_scale = self.safe_to_uint8(ifft0, ifft1)
        t_convert = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        num_iterations = self.fft_corrected_pol0.shape[0]

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE IFFT (CreateFilteredTimeDomainSignal) ===")
        print(f" Iterazioni elaborate:              {num_iterations}")
        print(f" Scala Ottimale Calcolata:           {optimal_scale:.6f}")
        print(f" 1. Calcolo IFFT (Pol0 + Pol1):     {t_ifft:.3f} s")
        print(f" 2. Conversione a uint8 (safe_clip): {t_convert*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE IFFT:                 {t_total:.3f} s")
        print("==========================================================\n")

        # Emissione del segnale con r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1 e il fattore di scala
        self.result_signal_filtered_time_domain.emit(r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1, optimal_scale)

        self.update_statusbar_signal.emit('Segnale tempo-dominio ricostruito con successo.')
        self.update_button_signal.emit(True)
        self.update_progressbar_value.emit(100)




class CreateFilteredTimeDomainSignal(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    result_signal_filtered_time_domain = pyqtSignal(object, object, object, object)

    def __init__(self, fft_corrected_pol0, fft_corrected_pol1, outliers_index_pol0, outliers_index_pol1, parent=None):
        super().__init__(parent)
        self.fft_corrected_pol0 = fft_corrected_pol0
        self.fft_corrected_pol1 = fft_corrected_pol1
        self.outliers_index_pol0 = outliers_index_pol0
        self.outliers_index_pol1 = outliers_index_pol1


    
    def safe_to_uint8(self, time_domain_complex, target_sigma_bins=32.0):
        """
        Calcola la scala dinamica ottimale basata sulla deviazione standard (RMS) 
        invece del picco isolato, garantendo la massima dinamica senza abbassare la potenza.
        """
        # 1. Calcolo della deviazione standard del segnale nel tempo
        std_real = np.std(time_domain_complex.real)
        std_imag = np.std(time_domain_complex.imag)
        current_std = max(std_real, std_imag)

        # 2. Vogliamo che la std del segnale occupi circa 32 canali uint8 (così 4*sigma = 128, perfetto per [0, 255])
        if current_std > 0:
            scale_factor = target_sigma_bins / current_std
            # Limite di sicurezza superiore per evitare sovra-amplificazione su segnali piatti
            scale_factor = min(scale_factor, 0.95)
        else:
            scale_factor = 0.9

        print('Scale Factor:', scale_factor)

        # 3. Conversione a uint8
        i_scaled = np.clip(time_domain_complex.real * scale_factor + 128.5, 0, 255).astype(np.uint8)
        q_scaled = np.clip(time_domain_complex.imag * scale_factor + 128.5, 0, 255).astype(np.uint8)
            
        return i_scaled, q_scaled


    def safe_to_uint8_old(self, time_domain_complex, margin=127.0):
        """
        Calcola il massimo valore del segnale complesso temporale e applica 
        il fattore di scala dinamico ottimale per sfruttare il 100% della dinamica 
        a 8-bit senza produrre clipping.
        """
        # 1. Trova il picco massimo assoluto tra parte Reale e Immaginaria
        max_real = np.max(np.abs(time_domain_complex.real))
        max_imag = np.max(np.abs(time_domain_complex.imag))
        peak_val = max(max_real, max_imag)

        # 2. Calcola il fattore di scala massimo teorico per evitare la saturazione
        if peak_val > 0:
            scale_factor = margin / peak_val
        else:
            scale_factor = 1.0

        # 3. Conversione dinamica senza clipping
        i_scaled = np.clip(time_domain_complex.real * scale_factor + 128.5, 0, 255).astype(np.uint8)
        q_scaled = np.clip(time_domain_complex.imag * scale_factor + 128.5, 0, 255).astype(np.uint8)

        print('Scale Factor:', scale_factor)

        return i_scaled, q_scaled

    def run(self):
        t_start_total = time.perf_counter()

        self.update_button_signal.emit(False)
        self.update_statusbar_signal.emit('Ricostruzione del segnale tempo-variante (8-bit uint)...')

        # 1. IFFT vettoriale su asse 1
        t0 = time.perf_counter()
        ifft0 = np.fft.ifft(self.fft_corrected_pol0, axis=1)
        ifft1 = np.fft.ifft(self.fft_corrected_pol1, axis=1)
        t_ifft = time.perf_counter() - t0

        # 2. Conversione sicura a uint8 senza wrap-around e senza clipping
        t0 = time.perf_counter()
        r_uint8_0, i_uint8_0 = self.safe_to_uint8(ifft0)
        r_uint8_1, i_uint8_1 = self.safe_to_uint8(ifft1)
        t_convert = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        num_iterations = self.fft_corrected_pol0.shape[0]

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE IFFT (CreateFilteredTimeDomainSignal) ===")
        print(f" Iterazioni elaborate:              {num_iterations}")
        print(f" 1. Calcolo IFFT (Pol0 + Pol1):     {t_ifft:.3f} s")
        print(f" 2. Conversione a uint8 (safe_clip): {t_convert*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE IFFT:                 {t_total:.3f} s")
        print("==========================================================\n")

        self.result_signal_filtered_time_domain.emit(r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1)

        self.update_statusbar_signal.emit('Segnale tempo-dominio ricostruito con successo.')
        self.update_button_signal.emit(True)
        self.update_progressbar_value.emit(100)



        

    
    
    def get_outliers(self, threshold_1, spectrum_pol1, threshold_2, spectrum_pol2):

        # Apply a median filter to get a smoothed version
        smooth_power_pol0 = medfilt(self.psd_pol0, kernel_size=5)
        # Identify outliers where power is way above the smoothed estimate
        threshold = 3  # tweak as needed
        outliers_mask = self.psd_pol0 > threshold * smooth_power_pol0
        #print(outliers)
        # Get indices of the outliers
        outlier_indices = np.where(outliers_mask)[0]

        print("Outlier indices:", outlier_indices)
