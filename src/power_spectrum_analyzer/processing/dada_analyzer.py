import numpy as np
import time
import concurrent.futures
import os

#from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import pyqtSignal, QThread

from scipy.signal import medfilt


import os
import concurrent.futures

import scipy.fft


class CreateRawSpectrum(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    result_signal_psd = pyqtSignal(object, object, object, object)

    def __init__(self, iterations, npoints, nchan, data, 
                 enable_chunking=True, max_iterations=156250, num_blocks=1, parent=None):
        super().__init__(parent)

        self.iterations = iterations
        self.npoints = npoints
        self.nchan = nchan
        self.data = data
        
        self.enable_chunking = enable_chunking
        self.max_iterations = max_iterations
        self.num_blocks = num_blocks

    def run(self):
        t_start_total = time.perf_counter()

        # Determina le iterazioni effettive da elaborare
        if self.enable_chunking and self.iterations > self.max_iterations:
            processed_iters = self.max_iterations
            mode_str = f"SUBSET CONTIGUO ({processed_iters} iterazioni)"
        else:
            processed_iters = self.iterations
            mode_str = f"FULL ({processed_iters} iterazioni)"

        self.update_statusbar_signal.emit(f'FFT calculation started [{mode_str}]...')    
        self.update_button_signal.emit(False)     

        rows_per_step = self.npoints  # 1024

        # 1. PREPARAZIONE VETTORIZZATA ISTANTANEA (ZERO COPY / ZERO INTERMEDII)
        t0 = time.perf_counter()
        # View a 3D senza copiare dati in RAM
        raw_subset = self.data[:processed_iters * rows_per_step].reshape(processed_iters, rows_per_step, 4)

        # Conversione diretta in numeri complessi float32 in un unico passaggio in memoria
        # I/Q sono centrati sottraendo 128.0 (128.0 + 128.0j sul numero complesso)
        complex_pol0 = (raw_subset[:, :, 0] + 1j * raw_subset[:, :, 1]).astype(np.complex64)
        complex_pol0 -= (128.0 + 128.0j)

        complex_pol1 = (raw_subset[:, :, 2] + 1j * raw_subset[:, :, 3]).astype(np.complex64)
        complex_pol1 -= (128.0 + 128.0j)
        t_prep = time.perf_counter() - t0

        # 2. CALCOLO FFT MULTITHREADING (scipy.fft con 4 o più core)
        t0 = time.perf_counter()
        # workers=4 impiega 4 core CPU. Se vuoi usare tutti i core logici disponibili usa workers=-1
        fft_array_pol0 = scipy.fft.fft(complex_pol0, axis=1, workers=4)
        fft_array_pol1 = scipy.fft.fft(complex_pol1, axis=1, workers=4)
        t_fft = time.perf_counter() - t0

        # 3. SPETTRO DI POTENZA (PSD) E MEDIA INTEGRATA OTTIMIZZATA
        t0 = time.perf_counter()
        self.update_statusbar_signal.emit('FFT calculation completed. Squaring coefficients...')
        self.update_progressbar_range.emit(0, 0)

        # np.abs(arr)**2 in NumPy per array complessi è implementato nativamente in C 
        # ed è più veloce di arr.real**2 + arr.imag**2
        pol0_psd_tot = np.mean(np.abs(fft_array_pol0)**2, axis=0)
        pol1_psd_tot = np.mean(np.abs(fft_array_pol1)**2, axis=0)
        t_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI E BENCHMARK ---
        print("\n=== BENCHMARK SPETTRO GREZZO OTTIMIZZATO (CreateRawSpectrum) ===")
        print(f" Modalita Esecuzione:                {mode_str}")
        print(f" Iterazioni Totali File DADA:        {self.iterations}")
        print(f" Iterazioni Realmente Elaborate:    {processed_iters} ({(processed_iters/self.iterations)*100:.1f}% dei dati)")
        print(f" 1. Preparazione & Complex Direct:  {t_prep*1000:.2f} ms")
        print(f" 2. Calcolo FFT Multithread (scipy): {t_fft:.3f} s")
        print(f" 3. Calcolo PSD e Integrazione Media:{t_psd*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE SPETTRO GREZZO:        {t_total:.3f} s")
        print("==========================================================\n")

        # Store results
        self.fft_array_pol0 = fft_array_pol0
        self.fft_array_pol1 = fft_array_pol1
        self.pol0_psd_tot = pol0_psd_tot
        self.pol1_psd_tot = pol1_psd_tot

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(processed_iters)

        self.result_signal_psd.emit(self.pol0_psd_tot, self.pol1_psd_tot, fft_array_pol0, fft_array_pol1)

        self.update_button_signal.emit(True)  
        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)





class CreateRawSpectrum___(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    result_signal_psd = pyqtSignal(object, object, object, object)

    def __init__(self, iterations, npoints, nchan, data, 
                 enable_chunking=True, max_iterations=156250, num_blocks=1, parent=None):
        super().__init__(parent)

        self.iterations = iterations
        self.npoints = npoints
        self.nchan = nchan
        self.data = data
        
        # --- PARAMETRI CHUNKING DINAMICO ---
        self.enable_chunking = enable_chunking
        self.max_iterations = max_iterations
        self.num_blocks = num_blocks


    def _extract_subset(self, rows_per_step):
        """
        Estrae il subset contiguo esatto richiesto dall'utente (es. prime N iterazioni).
        """
        actual_iters = min(self.iterations, self.max_iterations if self.enable_chunking else self.iterations)
        subset_data = self.data[:actual_iters * rows_per_step].reshape(actual_iters, rows_per_step, 4)
        return subset_data, actual_iters



    def _extract_strided_blocks__(self, rows_per_step):
        """
        Estrae num_blocks distribuiti uniformemente lungo l'INTERO file DADA
        (es. da 0 a 156.000 iterazioni).
        """
        total_iters = self.iterations
        
        if total_iters <= self.max_iterations:
            subset_data = self.data[:total_iters * rows_per_step].reshape(total_iters, rows_per_step, 4)
            return subset_data, total_iters

        block_size = self.max_iterations // self.num_blocks
        
        # Genera i punti di partenza equispaziati tra 0 e l'ultimo blocco possibile dell'INTERO file
        starts = np.linspace(0, total_iters - block_size, self.num_blocks, dtype=int)
        
        selected_chunks = []
        for start in starts:
            end = start + block_size
            chunk = self.data[start * rows_per_step : end * rows_per_step].reshape(block_size, rows_per_step, 4)
            selected_chunks.append(chunk)
            
        subset_data = np.concatenate(selected_chunks, axis=0)
        actual_iters = subset_data.shape[0]
        
        return subset_data, actual_iters





    def _extract_strided_blocks_old(self, rows_per_step):
        """
        Estrae num_blocks distribuiti uniformemente lungo l'osservazione DADA.
        Ritorna la matrice data_reshaped contenente solo le iterazioni selezionate.
        """
        total_iters = self.iterations
        
        if total_iters <= self.max_iterations:
            # Se le iterazioni totali sono già poche, prendiamo tutto
            subset_data = self.data[:total_iters * rows_per_step].reshape(total_iters, rows_per_step, 4)
            return subset_data, total_iters

        block_size = self.max_iterations // self.num_blocks
        starts = np.linspace(0, total_iters - block_size, self.num_blocks, dtype=int)
        
        selected_chunks = []
        for start in starts:
            end = start + block_size
            chunk = self.data[start * rows_per_step : end * rows_per_step].reshape(block_size, rows_per_step, 4)
            selected_chunks.append(chunk)
            
        # Concateniamo i blocchi campionati in un'unica matrice
        subset_data = np.concatenate(selected_chunks, axis=0)
        actual_iters = subset_data.shape[0]
        
        return subset_data, actual_iters


    def run(self):
        t_start_total = time.perf_counter()

        # Determina le iterazioni effettive da elaborare (subset contiguo iniziale)
        if self.enable_chunking and self.iterations > self.max_iterations:
            processed_iters = self.max_iterations
            mode_str = f"SUBSET CONTIGUO ({processed_iters} iterazioni)"
        else:
            processed_iters = self.iterations
            mode_str = f"FULL ({processed_iters} iterazioni)"

        self.update_statusbar_signal.emit(f'FFT calculation started [{mode_str}]...')   
        self.update_button_signal.emit(False)     

        rows_per_step = self.npoints  # 1024

        # 1. Reshape e centraggio a float32 sul solo subset contiguo selezionato
        t0 = time.perf_counter()
        data = self.data[:processed_iters * rows_per_step].reshape(processed_iters, rows_per_step, 4)
        
        # Estrazione I/Q e centraggio su 128
        i0 = data[:, :, 0].astype(np.float32) - 128.0
        q0 = data[:, :, 1].astype(np.float32) - 128.0
        i1 = data[:, :, 2].astype(np.float32) - 128.0
        q1 = data[:, :, 3].astype(np.float32) - 128.0
        t_prep = time.perf_counter() - t0

        # 2. Costruzione del segnale complesso
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
        t0 = time.perf_counter()
        self.update_statusbar_signal.emit('FFT calculation completed. Squaring coefficients...')
        self.update_progressbar_range.emit(0, 0)

        pol0_psd_tot = np.mean(fft_array_pol0.real**2 + fft_array_pol0.imag**2, axis=0)
        pol1_psd_tot = np.mean(fft_array_pol1.real**2 + fft_array_pol1.imag**2, axis=0)
        t_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI E BENCHMARK ---
        print("\n=== BENCHMARK SPETTRO GREZZO (CreateRawSpectrum) ===")
        print(f" Modalita Esecuzione:                {mode_str}")
        print(f" Iterazioni Totali File DADA:        {self.iterations}")
        print(f" Iterazioni Realmente Elaborate:    {processed_iters} ({(processed_iters/self.iterations)*100:.1f}% dei dati)")
        print(f" 1. Preparazione Dati e Centraggio: {t_prep:.3f} s")
        print(f" 2. Costruzione Segnale Complesso:   {t_complex*1000:.2f} ms")
        print(f" 3. Calcolo FFT (Pol0 + Pol1):       {t_fft:.3f} s")
        print(f" 4. Calcolo PSD e Integrazione Media:{t_psd*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE SPETTRO GREZZO:        {t_total:.3f} s")
        print("==========================================================\n")

        # Store results
        self.fft_array_pol0 = fft_array_pol0
        self.fft_array_pol1 = fft_array_pol1
        self.pol0_psd_tot = pol0_psd_tot
        self.pol1_psd_tot = pol1_psd_tot

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(processed_iters)

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

    # --- NUOVO SEGNALE DEDICATO PER I DATI RIPARATI NEL TEMPO ---
    result_signal_time_clean = pyqtSignal(object, object)


    def __init__(self, raw_data, total_iterations, npoints, 
                 outliers_indices_pol0, outliers_indices_pol1, fft_raw_pol0=None, parent=None):
        super().__init__(parent)

        self.raw_data = raw_data
        self.total_iterations = total_iterations
        self.npoints = npoints
        self.outliers_indices_pol0 = outliers_indices_pol0
        self.outliers_indices_pol1 = outliers_indices_pol1
        
        # Determina la dimensione reale del SUBSET da elaborare
        if fft_raw_pol0 is not None:
            self.subset_iterations = fft_raw_pol0.shape[0]
        else:
            self.subset_iterations = total_iterations

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
        if not outliers_indices:
            return fft_data.copy()

        num_iterations = fft_data.shape[0]
        num_outliers = len(outliers_indices)

        safe_mags = np.abs(fft_data[:, safe_channels])
        m = np.mean(safe_mags, axis=0)
        s = np.std(safe_mags, axis=0)

        # Generatore PCG64 ultra-veloce a 32-bit
        rng = np.random.default_rng()
        z_random = rng.standard_normal((num_iterations, num_outliers), dtype=np.float32)

        new_mags = np.maximum(m + z_random * s, 1e-9)
        orig_mags = np.abs(fft_data[:, outliers_indices]) + 1e-9

        clean_fft = fft_data.copy()
        clean_fft[:, outliers_indices] *= (new_mags / orig_mags)

        return clean_fft


    
    def _repair_time_domain_vectorized(self, fft_data, threshold=3.0):
        """
        Riparazione RFI temporale via Random Donor Subset.
        Seleziona vettori FFT reali da un pool di iterazioni sani:
        - Nessun artefatto fisso/a strisce nel Waterfall 2D.
        - Serie Temporale 1D piatta e naturale.
        - Vettorizzato e ultra-veloce.
        """
        clean_fft = fft_data.copy()
        n_iters, n_chan = fft_data.shape

        # 1. Calcolo rapido della Serie Temporale Integrata 1D
        power_2d = clean_fft.real**2 + clean_fft.imag**2
        time_series = np.mean(power_2d, axis=1)

        med_ts = np.median(time_series)
        mad_ts = np.median(np.abs(time_series - med_ts)) + 1e-9
        mod_z_ts = 0.6745 * (time_series - med_ts) / mad_ts

        # 2. Separazione tra iterazioni anomale e Pool di Donatori Sani
        bad_time_mask = np.abs(mod_z_ts) > threshold
        bad_time_indices = np.where(bad_time_mask)[0]
        
        # Consideriamo "sani" solo i punti strettamente sotto soglia (|Z| <= threshold)
        good_time_indices = np.where(~bad_time_mask)[0]

        print(f"\n[Random Donor Subset QA] Iterazioni corrotte: {len(bad_time_indices)} | Pool donatori sani: {len(good_time_indices)}")

        if len(bad_time_indices) > 0 and len(good_time_indices) > 0:
            # 3. Estrazione casuale dal subset sano per ciascuna riga infetta
            rng = np.random.default_rng()
            random_donor_indices = rng.choice(good_time_indices, size=len(bad_time_indices), replace=True)

            # 4. Sostituzione vettorizzata a blocchi (Zero cicli Python)
            clean_fft[bad_time_indices, :] = clean_fft[random_donor_indices, :]

        return clean_fft
     
       
    def _repair_time_domain_vectorized___(self, fft_data, threshold=3.0):
        """
        Riparazione RFI temporale via Reference Block Copy (Donor fissa iniziale).
        Usa le prime iterazioni sane come profilo di riferimento immutabile:
        - Velocità di esecuzione ISTANTANEA.
        - Zero artefatti spettrali.
        - Serie Temporale 1D perfettamente pulita.
        """
        clean_fft = fft_data.copy()
        n_iters, n_chan = fft_data.shape

        # 1. Calcolo veloce della Serie Temporale Integrata 1D
        power_2d = clean_fft.real**2 + clean_fft.imag**2
        time_series = np.mean(power_2d, axis=1)

        med_ts = np.median(time_series)
        mad_ts = np.median(np.abs(time_series - med_ts)) + 1e-9
        mod_z_ts = 0.6745 * (time_series - med_ts) / mad_ts

        # 2. Individuazione delle iterazioni anomale
        bad_time_mask = np.abs(mod_z_ts) > threshold
        bad_time_indices = np.where(bad_time_mask)[0]

        print(f"\n[Fixed Donor QA] Iterazioni anomale rimosse: {len(bad_time_indices)} / {n_iters}")

        if len(bad_time_indices) > 0:
            # PRENDIAMO COME RIFERIMENTO SANO LA PRIMA ITERAZIONE (o la media delle prime 10)
            # Se la prima iterazione fosse anomala, prendiamo la prima riga sana disponibile
            first_good_idx = 0
            if bad_time_mask[0]:
                good_candidates = np.where(~bad_time_mask)[0]
                first_good_idx = good_candidates[0] if len(good_candidates) > 0 else 0

            # Vettore complessi FFT di riferimento (Prima iterazione sana)
            reference_vector = clean_fft[first_good_idx, :]

            # SOSTITUZIONE ISTANTANEA A BLOCCO DI TUTTE LE ITERAZIONI ANOMALE
            clean_fft[bad_time_indices, :] = reference_vector

        return clean_fft





    def _repair_time_domain_vectorized__(self, fft_data, threshold=3.0):
        """
        Riparazione RFI temporali 2D mediante Noise Replacement coerente.
        Elimina gli spike positivi senza creare buchi o spike negativi nella serie temporale.
        """
        clean_fft = fft_data.copy()
        power_2d = clean_fft.real**2 + clean_fft.imag**2

        # 1. Identifichiamo le iterazioni globalmente infette
        time_series = np.mean(power_2d, axis=1)
        med_ts = np.median(time_series)
        mad_ts = np.median(np.abs(time_series - med_ts)) + 1e-9
        mod_z_ts = 0.6745 * (time_series - med_ts) / mad_ts

        # Iterazioni sane da cui estrarre la statistica di rumore pura
        good_indices = np.where(mod_z_ts <= threshold)[0]
        bad_indices = np.where(mod_z_ts > threshold)[0]

        print(f"\n[Time-Domain QA] Iterazioni corrotte rimosse e rigenerate: {len(bad_indices)} / {fft_data.shape[0]}")

        if len(bad_indices) > 0 and len(good_indices) > 0:
            # Calcolo media e deviazione standard reali/immaginarie dai canali e dai tempi sani
            mean_real = np.mean(clean_fft[good_indices, :].real, axis=0)
            std_real = np.std(clean_fft[good_indices, :].real, axis=0)
            mean_imag = np.mean(clean_fft[good_indices, :].imag, axis=0)
            std_imag = np.std(clean_fft[good_indices, :].imag, axis=0)

            # Generazione del rumore sostitutivo Gaussiano
            rng = np.random.default_rng()
            rand_real = rng.normal(mean_real, std_real, size=(len(bad_indices), fft_data.shape[1])).astype(np.float32)
            rand_imag = rng.normal(mean_imag, std_imag, size=(len(bad_indices), fft_data.shape[1])).astype(np.float32)

            # Iniezione del rumore sano al posto dei dati infetti
            clean_fft[bad_indices, :] = rand_real + 1j * rand_imag

        return clean_fft


  


    def run(self):
        t_start_total = time.perf_counter()
        max_workers = 4

        subset_iters = self.subset_iterations

        self.update_statusbar_signal.emit(f'Calcolo FFT e filtraggio su SUBSET ({subset_iters} iterazioni)...')   
        self.update_button_signal.emit(False)  
        self.update_progressbar_range.emit(0, 0)

        # 1. Ricerca canali sani
        t0 = time.perf_counter()
        channels_pol0 = self._find_safe_channels_fast(self.outliers_indices_pol0, min_dist=5)
        channels_pol1 = self._find_safe_channels_fast(self.outliers_indices_pol1, min_dist=5)
        t_safe_channels = time.perf_counter() - t0

        # 2. Calcolo FFT sul SUBSET
        t0 = time.perf_counter()
        rows_per_step = self.npoints
        subset_data = self.raw_data[:subset_iters * rows_per_step].reshape(subset_iters, rows_per_step, 4)
        
        i0 = subset_data[:, :, 0].astype(np.float32) - 128.0
        q0 = subset_data[:, :, 1].astype(np.float32) - 128.0
        i1 = subset_data[:, :, 2].astype(np.float32) - 128.0
        q1 = subset_data[:, :, 3].astype(np.float32) - 128.0
        
        full_fft_pol0 = scipy.fft.fft(i0 + 1j * q0, axis=1, workers=max_workers)
        full_fft_pol1 = scipy.fft.fft(i1 + 1j * q1, axis=1, workers=max_workers)
        t_full_fft = time.perf_counter() - t0

        # 3. RIPARAZIONE VETTORIALE (1° Frequenza -> 2° Tempo)
        t0 = time.perf_counter()
        # Passo A: Riparazione Outlier in Frequenza
        clean_fft_pol0 = self._repair_polarization_vectorized(full_fft_pol0, self.outliers_indices_pol0, channels_pol0)
        clean_fft_pol1 = self._repair_polarization_vectorized(full_fft_pol1, self.outliers_indices_pol1, channels_pol1)
        
        # Passo B: Riparazione Dominio del Tempo (applicata direttamente sulla matrice già filtrata in frequenza)
        time_clean_fft_pol0 = self._repair_time_domain_vectorized(clean_fft_pol0, threshold=3.0)
        time_clean_fft_pol1 = self._repair_time_domain_vectorized(clean_fft_pol1, threshold=3.0)
        t_repair = time.perf_counter() - t0

        # 4. CREAZIONE MATRICE POTENZA 2D RIPULITA DA PASSARE ALLA GUI
        t0 = time.perf_counter()
        # Matrice 2D di potenza (num_iterazioni x nchan) ripulita sia in Frequenza che nel Tempo
        time_repaired_psd_2d_pol0 = time_clean_fft_pol0.real**2 + time_clean_fft_pol0.imag**2
        time_repaired_psd_2d_pol1 = time_clean_fft_pol1.real**2 + time_clean_fft_pol1.imag**2

        # Spettri 1D mediati dai dati riparati nel tempo
        psd0_tot = np.mean(time_repaired_psd_2d_pol0, axis=0)
        psd1_tot = np.mean(time_repaired_psd_2d_pol1, axis=0)
        t_reconstruct_psd = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total

        # --- TRACCIAMENTO TEMPI INTEGRAZIONE E BENCHMARK ---
        print("\n=== TEMPI ESECUZIONE PULIZIA INTEGRALE (CreateFilteredSpectrum - Multithread) ===")
        print(f" Iterazioni Totali Riparate:         {subset_iters}")
        print(f" CPU Worker Allocati:                {max_workers}")
        print(f" 1. Ricerca Canali Sani:             {t_safe_channels*1000:.2f} ms")
        print(f" 2. FFT Multithread ({max_workers} worker): {t_full_fft:.3f} s")
        print(f" 3. Replacement Vettoriale (Freq+Time): {t_repair*1000:.2f} ms")
        print(f" 4. Calcolo PSD Finale:              {t_reconstruct_psd*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE PULIZIA INTEGRALE:     {t_total:.3f} s")
        print("==========================================================\n")

        # --- EMISSIONE SEGNALI CON LA NUOVA MATRICE RIPULITA ---
        self.update_statusbar_signal.emit('Filtered Power Spectrum ready.')
        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)

        # Invia la matrice FFT complessa filtrata sia in frequenza che nel tempo
        self.result_signal_psd.emit(psd0_tot, psd1_tot, time_clean_fft_pol0, time_clean_fft_pol1)

        # Invia le matrici 2D di potenza reali al segnale dedicato di diagnostica
        if hasattr(self, 'result_signal_time_clean'):
            self.result_signal_time_clean.emit(time_repaired_psd_2d_pol0, time_repaired_psd_2d_pol1)

        self.update_button_signal.emit(True)
    


class DetectOutliers(QThread):

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    outliers_signal_psd = pyqtSignal(object, object)

    def __init__(self, power_spectrum_pol0, chunk_size_pol0, threshold_pol0,
                 power_spectrum_pol1, chunk_size_pol1, threshold_pol1, enable_flattening=True, parent=None):
        super().__init__(parent)

        self.power_spectrum_pol0 = np.array(power_spectrum_pol0)
        self.chunk_size_pol0 = int(chunk_size_pol0)
        self.threshold_pol0 = float(threshold_pol0)

        self.power_spectrum_pol1 = np.array(power_spectrum_pol1)
        self.chunk_size_pol1 = int(chunk_size_pol1)
        self.threshold_pol1 = float(threshold_pol1)

        # Flag per abilitare/disabilitare l'appiattimento preventivo dello spettro 1D
        self.enable_flattening = enable_flattening  # <-- SALVATAGGIO PARAMETRO


     
    def _flatten_spectrum(self, spectrum, window_len=101):
        """
        Metodo ad-hoc per l'appiattimento chirurgico dello Spettro 1D.
        Utilizza un filtro mediano ad ampia finestra (default 101 canali) per stimare 
        la curvatura dell'hardware (roll-off) senza intaccare o cancellare gli spike RFI stretti.
        Ritorna lo spettro normalizzato attorno a 1.0.
        """
        # Dimensione finestra forzata a valore dispari
        size = window_len if window_len % 2 != 0 else window_len + 1
        
        # Stima della baseline morbida del ricevitore
        baseline = medfilt(spectrum, kernel_size=size)
        baseline = np.maximum(baseline, 1e-9)
        
        # Spettro appiattito centrato attorno a 1.0
        return spectrum / baseline



    
    def _find_outliers(self, spectrum, chunk_size, threshold):
        """
        Rilevamento chirurgico ed adattivo degli outlier spettrali (RFI):
        - Rileva picchi principali tramite Modified Z-Score sul gradiente/residuo.
        - Applica un floor di sicurezza al MAD (safe_mad) per evitare falsi positivi su rumore integrato.
        - Espansione ad ampiezza controllata (shoulder_threshold) per isolare le sole RFI reali.
        """
        spectrum = np.array(spectrum, dtype=np.float64)
        nchan = len(spectrum)
        guard_band = 20  # Ignora bordi estremi della banda

        # 1. Stima e eventuale appiattimento della baseline
        if self.enable_flattening:
            flat_spec = self._flatten_spectrum(spectrum, window_len=101)
        else:
            baseline = medfilt(spectrum, kernel_size=31)
            baseline = np.maximum(baseline, 1e-9)
            flat_spec = spectrum / baseline

        # =====================================================================
        # STAMPA A VIDEO DEI 1024 VALORI (spectrum / baseline)
        # =====================================================================
        print("\n" + "="*80)
        print(" DIAGNOSTICA SPETTRO APPIATTITO (flat_spec = spectrum / baseline)")
        print(f" Statistica 1024 Canali -> Media: {np.mean(flat_spec):.4f} | Mediana: {np.median(flat_spec):.4f} | Min: {np.min(flat_spec):.4f} | Max: {np.max(flat_spec):.4f}")
        print("="*80)
        
        # Stampa dei 1024 valori organizzati in righe da 16 canali
        for i in range(0, nchan, 16):
            chunk_vals = " ".join([f"{val:6.3f}" for val in flat_spec[i:i+16]])
            print(f"Canali [{i:4d}-{i+15:4d}]:  {chunk_vals}")
        print("="*80 + "\n")
        # =====================================================================

        # 2. Modified Z-Score con floor di sicurezza sul MAD (safe_mad)
        med_val = np.median(flat_spec)
        mad_val = np.median(np.abs(flat_spec - med_val)) + 1e-12
        
        # Floor di sicurezza dell'1% (0.01) per evitare che deviazioni del rumore < 2% generino falsi positivi
        safe_mad = max(mad_val, 0.01)
        mod_z = 0.6745 * (flat_spec - med_val) / safe_mad

        # Canali primari che superano la soglia specificata dall'utente
        primary_peaks = np.where(mod_z > threshold)[0]

        if len(primary_peaks) == 0:
            return []

        infected_channels = set()
        
        # Soglia di rientro controllata: ferma l'espansione al rientro nel rumore naturale (Z <= 2.0)
        shoulder_threshold = max(2.0, threshold * 0.5)

        # 3. Espansione dinamica adattiva per ciascun picco trovato
        for peak in primary_peaks:
            if guard_band <= peak < nchan - guard_band:
                infected_channels.add(peak)

                # Espansione a SINISTRA
                left = peak - 1
                while left >= guard_band and mod_z[left] > shoulder_threshold:
                    infected_channels.add(left)
                    left -= 1

                # Espansione a DESTRA
                right = peak + 1
                while right < nchan - guard_band and mod_z[right] > shoulder_threshold:
                    infected_channels.add(right)
                    right += 1

        return sorted(list(infected_channels))

 




    def _find_outliers_old(self, spectrum, chunk_size, threshold):
        """
        Rilevamento chirurgico ed adattivo degli outlier spettrali (RFI):
        - Rileva picchi principali tramite Modified Z-Score sul gradiente/residuo.
        - Per spike singoli di 1 canale: isola solo quel canale senza intaccare i vicini sani.
        - Per picchi complessi/multicanale: segue la discesa del profilo (shoulder)
          fino al rientro nella baseline sana (shoulder_threshold).
        """
        spectrum = np.array(spectrum, dtype=np.float64)
        nchan = len(spectrum)
        guard_band = 20  # Ignora bordi estremi della banda

        # 1. Stima e eventuale appiattimento della baseline
        if self.enable_flattening:
            # Richiama il metodo ad-hoc se il flag è True
            flat_spec = self._flatten_spectrum(spectrum, window_len=101)
        else:
            # Comportamento originale
            baseline = medfilt(spectrum, kernel_size=31)
            baseline = np.maximum(baseline, 1e-9)
            flat_spec = spectrum / baseline

        
        # =====================================================================
        # STAMPA A VIDEO DEI 1024 VALORI (spectrum / baseline)
        # =====================================================================
        print("\n" + "="*80)
        print(" DIAGNOSTICA SPETTRO APPIATTITO (flat_spec = spectrum / baseline)")
        print(f" Statistica 1024 Canali -> Media: {np.mean(flat_spec):.4f} | Mediana: {np.median(flat_spec):.4f} | Min: {np.min(flat_spec):.4f} | Max: {np.max(flat_spec):.4f}")
        print("="*80)
        
        # Stampa dei 1024 valori organizzati in righe da 16 canali per facilitare la lettura
        for i in range(0, nchan, 16):
            chunk_vals = " ".join([f"{val:6.3f}" for val in flat_spec[i:i+16]])
            print(f"Canali [{i:4d}-{i+15:4d}]:  {chunk_vals}")
        print("="*80 + "\n")
        # =====================================================================



        # 2. Modified Z-Score basato sulla deviazione robusta (MAD)
        med_val = np.median(flat_spec)
        mad_val = np.median(np.abs(flat_spec - med_val)) + 1e-12
        mod_z = 0.6745 * (flat_spec - med_val) / mad_val

        # Canali primari che superano la soglia specificata dall'utente
        primary_peaks = np.where(mod_z > threshold)[0]

        if len(primary_peaks) == 0:
            return []

        infected_channels = set()
        
        # Soglia per definire il rientro nel rumore sano durante la discesa (shoulder threshold)
        shoulder_threshold = min(1.5, threshold * 0.4)

        # 3. Espansione dinamica adattiva per ciascun picco trovato
        for peak in primary_peaks:
            if guard_band <= peak < nchan - guard_band:
                infected_channels.add(peak)

                # Espansione a SINISTRA
                left = peak - 1
                while left >= guard_band and mod_z[left] > shoulder_threshold:
                    infected_channels.add(left)
                    left -= 1

                # Espansione a DESTRA
                right = peak + 1
                while right < nchan - guard_band and mod_z[right] > shoulder_threshold:
                    infected_channels.add(right)
                    right += 1

        return sorted(list(infected_channels))




   
    


    def run(self):
        self.update_statusbar_signal.emit('Inspecting the Power Spectrum for outliers...')
        self.update_button_signal.emit(False)
        self.update_progressbar_range.emit(0, 0)

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

        print("\n=== OUTLIER DETECTION RESULTS ===")
        print(f" Outliers POL0 ({len(outlier_indices_pol0)} canali): {outlier_indices_pol0}")
        print(f" Outliers POL1 ({len(outlier_indices_pol1)} canali): {outlier_indices_pol1}")
        print("=================================\n")

        self.outliers_signal_psd.emit(outlier_indices_pol0, outlier_indices_pol1)

        self.update_statusbar_signal.emit('Outlier detection completed.')
        self.update_button_signal.emit(True)
        self.update_progressbar_value.emit(100)






class DetectOutliers___(QThread): # without flattening

    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    update_progressbar_range = pyqtSignal(int, int)
    outliers_signal_psd = pyqtSignal(object, object)

    def __init__(self, power_spectrum_pol0, chunk_size_pol0, threshold_pol0,
                 power_spectrum_pol1, chunk_size_pol1, threshold_pol1, parent=None):
        super().__init__(parent)

        self.power_spectrum_pol0 = np.array(power_spectrum_pol0)
        self.chunk_size_pol0 = int(chunk_size_pol0)
        self.threshold_pol0 = float(threshold_pol0)

        self.power_spectrum_pol1 = np.array(power_spectrum_pol1)
        self.chunk_size_pol1 = int(chunk_size_pol1)
        self.threshold_pol1 = float(threshold_pol1)


    def _find_outliers(self, spectrum, chunk_size, threshold):
        """
        Rilevamento chirurgico ed adattivo degli outlier spettrali (RFI):
        - Rileva picchi principali tramite Modified Z-Score sul gradiente/residuo.
        - Per spike singoli di 1 canale: isola solo quel canale senza intaccare i vicini sani.
        - Per picchi complessi/multicanale: segue la discesa del profilo (shoulder)
          fino al rientro nella baseline sana (shoulder_threshold).
        """
        spectrum = np.array(spectrum, dtype=np.float64)
        nchan = len(spectrum)
        guard_band = 20  # Ignora bordi estremi della banda

        # 1. Stima della baseline con filtro mediano
        baseline = medfilt(spectrum, kernel_size=31)
        baseline = np.maximum(baseline, 1e-9)
        flat_spec = spectrum / baseline

        # 2. Modified Z-Score basato sulla deviazione robusta (MAD)
        med_val = np.median(flat_spec)
        mad_val = np.median(np.abs(flat_spec - med_val)) + 1e-12
        mod_z = 0.6745 * (flat_spec - med_val) / mad_val

        # Canali primari che superano la soglia specificata dall'utente (es. threshold = 3.0 - 4.5)
        primary_peaks = np.where(mod_z > threshold)[0]

        if len(primary_peaks) == 0:
            return []

        infected_channels = set()
        
        # Soglia per definire il rientro nel rumore sano durante la discesa (shoulder threshold)
        shoulder_threshold = min(1.5, threshold * 0.4)

        # 3. Espansione dinamica adattiva per ciascun picco trovato
        for peak in primary_peaks:
            if guard_band <= peak < nchan - guard_band:
                infected_channels.add(peak)

                # Espansione a SINISTRA: cammina finché i canali sono sopra il rumore di fondo
                left = peak - 1
                while left >= guard_band and mod_z[left] > shoulder_threshold:
                    infected_channels.add(left)
                    left -= 1

                # Espansione a DESTRA: cammina finché i canali sono sopra il rumore di fondo
                right = peak + 1
                while right < nchan - guard_band and mod_z[right] > shoulder_threshold:
                    infected_channels.add(right)
                    right += 1

        return sorted(list(infected_channels))
    




    def _find_outliers______(self, spectrum, chunk_size, threshold):
        """
        Rilevamento outlier basato sul gradiente locale (np.diff).
        Ignora la forma a trapezio della banda e rileva solo spike spettrali stretti (RFI).
        """
        spectrum = np.array(spectrum, dtype=np.float64)
        n = len(spectrum)
        
        # Ignora i bordi estremi (es. 20 canali a sx e dx)
        guard_band = 20
        
        # 1. Calcolo del gradiente tra canali adiacenti
        diff_spec = np.diff(spectrum)
        
        # 2. Mediana e MAD sulla derivata dello spettro
        med_diff = np.median(diff_spec)
        mad_diff = np.median(np.abs(diff_spec - med_diff)) + 1e-12

        # 3. Modified Z-Score sul gradiente
        mod_z_diff = 0.6745 * np.abs(diff_spec - med_diff) / mad_diff
        
        # 4. Individuazione dei bordi degli spike
        diff_outliers = np.where(mod_z_diff > threshold)[0]
        
        # Map sui canali originali (se il gradiente d è anomalo, il canale d o d+1 è infetto)
        suspect_channels = set()
        for idx in diff_outliers:
            if guard_band <= idx < n - guard_band:
                suspect_channels.add(idx)
                suspect_channels.add(idx + 1)

        # 5. Conferma locale: verifica che il canale sia un picco rispetto ai 10 vicini
        final_outliers = []
        for ch in sorted(suspect_channels):
            left = max(0, ch - 5)
            right = min(n, ch + 6)
            neighbors = np.concatenate([spectrum[left:ch], spectrum[ch+1:right]])
            
            if len(neighbors) > 0:
                local_med = np.median(neighbors)
                local_mad = np.median(np.abs(neighbors - local_med)) + 1e-12
                
                # Deve essere un picco reale rispetto alla linea di base circostante
                if (spectrum[ch] - local_med) / local_mad > threshold:
                    final_outliers.append(int(ch))

        return final_outliers

    def run(self):
        self.update_statusbar_signal.emit('Inspecting the Power Spectrum for outliers...')
        self.update_button_signal.emit(False)
        self.update_progressbar_range.emit(0, 0)

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

        print("\n=== OUTLIER DETECTION RESULTS ===")
        print(f" Outliers POL0 ({len(outlier_indices_pol0)} canali): {outlier_indices_pol0}")
        print(f" Outliers POL1 ({len(outlier_indices_pol1)} canali): {outlier_indices_pol1}")
        print("=================================\n")

        self.outliers_signal_psd.emit(outlier_indices_pol0, outlier_indices_pol1)

        self.update_statusbar_signal.emit('Outlier detection completed.')
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

    def safe_to_uint8(self, ifft0, ifft1, margin=127.0, max_allowed_overflow_pct=0.001):
        """
        Versione ad alte prestazioni: quantizza Pol0 e Pol1 simultaneamente 
        usando le viste dirette float32 ed evitando allocazioni di memoria duplicate.
        """
        # Vista memory-view a float32 per evitare l'allocazione pesante di np.abs()
        v0 = ifft0.view(np.float32)
        v1 = ifft1.view(np.float32)
        
        # 1. Picco massimo globale tra le 2 polarizzazioni
        peak_val = max(np.max(np.abs(v0)), np.max(np.abs(v1)))

        # 2. Conteggio esatto dei campioni che sforano il margine 127.0
        overflow_cnt_0 = np.count_nonzero(np.abs(v0) > margin)
        overflow_cnt_1 = np.count_nonzero(np.abs(v1) > margin)
        total_overflow_cnt = overflow_cnt_0 + overflow_cnt_1
        
        total_samples = v0.size + v1.size
        overflow_pct = (total_overflow_cnt / total_samples) * 100

        # 3. Decisione sulla scala
        if overflow_pct > max_allowed_overflow_pct:
            scale_factor = margin / peak_val
            action = f"Scala Ridotta a {scale_factor:.4f} per preservare {total_overflow_cnt} punti."
        else:
            scale_factor = 1.0
            action = f"Scala Mantenuta a 1.0 (Solo {total_overflow_cnt} punti clippati)."

        print(f"\n  [Quantizzazione uint8 - Diagnostica Totale]")
        print(f"  +- Picco Massimo Assoluto: {peak_val:.2f} (Soglia max: {margin})")
        print(f"  +- Campioni Fuori Scala:   Pol0={overflow_cnt_0} | Pol1={overflow_cnt_1} (Totale: {total_overflow_cnt})")
        print(f"  +- Percentuale Overflow:   {overflow_pct:.6f}%")
        print(f"  +- Decisione:              {action}")

        # 4. Estraggo Reali (pari) ed Immaginari (dispari) mantenendo la forma 2D (num_iterations, npoints)
        i_scaled_0 = np.clip(v0[:, 0::2] * scale_factor + 128.5, 0, 255).astype(np.uint8)
        q_scaled_0 = np.clip(v0[:, 1::2] * scale_factor + 128.5, 0, 255).astype(np.uint8)

        i_scaled_1 = np.clip(v1[:, 0::2] * scale_factor + 128.5, 0, 255).astype(np.uint8)
        q_scaled_1 = np.clip(v1[:, 1::2] * scale_factor + 128.5, 0, 255).astype(np.uint8)

        return i_scaled_0, q_scaled_0, i_scaled_1, q_scaled_1

    def run(self):
        t_start_total = time.perf_counter()

        # Limite impostato a 8 CPU worker per il multithreading
        max_workers = 4

        self.update_button_signal.emit(False)
        self.update_statusbar_signal.emit(f'Ricostruzione segnale tempo-variante (IFFT con {max_workers} CPU)...')

        # 1. IFFT vettoriale vettorizzata in parallelo con scipy (8 workers)
        t0 = time.perf_counter()
        ifft0 = scipy.fft.ifft(self.fft_corrected_pol0, axis=1, workers=max_workers)
        ifft1 = scipy.fft.ifft(self.fft_corrected_pol1, axis=1, workers=max_workers)
        t_ifft = time.perf_counter() - t0

        # 2. Quantizzazione sicura a uint8 ottimizzata in un unico passaggio
        t0 = time.perf_counter()
        r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1 = self.safe_to_uint8(ifft0, ifft1)
        t_convert = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start_total
        num_iterations = self.fft_corrected_pol0.shape[0]

        # --- TRACCIAMENTO TEMPI ---
        print("\n=== TEMPI ESECUZIONE IFFT (CreateFilteredTimeDomainSignal - 8 Workers) ===")
        print(f" Iterazioni elaborate:              {num_iterations}")
        print(f" CPU Worker Allocati:                {max_workers}")
        print(f" 1. Calcolo IFFT (Pol0 + Pol1):     {t_ifft:.3f} s")
        print(f" 2. Conversione a uint8 (safe_clip): {t_convert*1000:.2f} ms")
        print(f"----------------------------------------------------------")
        print(f" TEMPO TOTALE IFFT:                 {t_total:.3f} s")
        print("==========================================================\n")

        # Ritorna i 4 array 2D separati con forma esatta (156250, 1024) ciascuno
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
