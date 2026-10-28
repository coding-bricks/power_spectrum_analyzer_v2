import numpy as np
import time

#from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import pyqtSignal, QThread

from scipy.signal import medfilt

class CreateRawSpectrum(QThread):

    # Signal to notify the main thread to disable/enable the button in the interface
    update_button_signal = pyqtSignal(bool)
    # Signal to notify the main thread (UI) to update the status bar in the interface 
    update_statusbar_signal = pyqtSignal(str)
    # Signal to notify the main thread (UI) to update the progressbar current value
    update_progressbar_value = pyqtSignal(int)
    # Signal to notify the main thread (UI) to update the progressbar current range
    update_progressbar_range = pyqtSignal(int, int)
    # Signal to notify the main thread with the result of the processing
    result_signal_psd = pyqtSignal(object, object, object, object)
    

    def __init__(self, iterations, npoints, nchan, data, parent=None):
        super().__init__(parent)

        self.iterations = iterations
        self.npoints = npoints
        self.nchan = nchan
        self.data = data
        
    def run(self):

        self.update_statusbar_signal.emit('FFT calclulation started...')   
        self.update_button_signal.emit(False)     


        rows_per_step = self.npoints  # 1024
        iterations = self.iterations

        # Take only the required portion of data
        data = self.data[:iterations * rows_per_step]

        # Reshape into (iterations, 1024, 4)
        data = data.reshape(iterations, rows_per_step, 4)

        # Convert uint8 ? centered int16
        data = data.astype(np.int16) - 128

        # Split polarizations
        real_pol0 = data[:, :, 0].astype(np.float32)
        imag_pol0 = data[:, :, 1].astype(np.float32)

        real_pol1 = data[:, :, 2].astype(np.float32)
        imag_pol1 = data[:, :, 3].astype(np.float32)

        # Build complex signals
        complex_signal_pol0 = real_pol0 + 1j * imag_pol0
        complex_signal_pol1 = real_pol1 + 1j * imag_pol1

        # FFT along frequency axis (axis=1)
        fft_array_pol0 = np.fft.fft(complex_signal_pol0, axis=1)
        fft_array_pol1 = np.fft.fft(complex_signal_pol1, axis=1)

        # Power spectrum (|FFT|^2)
        power_spectrum_pol0 = np.abs(fft_array_pol0) ** 2
        power_spectrum_pol1 = np.abs(fft_array_pol1) ** 2

        # Average spectrum across iterations
        self.update_statusbar_signal.emit('FFT calclulation completed. Squaring the FFT coefficients...')
        self.update_progressbar_range.emit(0, 0)
        pol0_psd_tot = power_spectrum_pol0.mean(axis=0)
        pol1_psd_tot = power_spectrum_pol1.mean(axis=0)

        # Store results
        self.fft_array_pol0 = fft_array_pol0
        self.fft_array_pol1 = fft_array_pol1
        self.power_spectrum_pol0 = power_spectrum_pol0
        self.power_spectrum_pol1 = power_spectrum_pol1
        self.pol0_psd_tot = pol0_psd_tot
        self.pol1_psd_tot = pol1_psd_tot

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(self.iterations)

        self.result_signal_psd.emit(self.pol0_psd_tot, self.pol1_psd_tot, fft_array_pol0, fft_array_pol1)

        self.update_button_signal.emit(True)  

        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100)
        
        #return self.pol0_psd_tot, self.pol1_psd_tot


class CreateFilteredSpectrum_new(QThread):

    # Signal to notify the main thread to disable/enable the button in the interface
    update_button_signal = pyqtSignal(bool)
    # Signal to notify the main thread (UI) to update the status bar in the interface 
    update_statusbar_signal = pyqtSignal(str)
    # Signal to notify the main thread (UI) to update the progressbar current value
    update_progressbar_value = pyqtSignal(int)
     # Signal to notify the main thread (UI) to update the progressbar current range
    update_progressbar_range = pyqtSignal(int, int)
    # Signal to notify the main thread with the result of the processing
    result_signal_psd = pyqtSignal(object, object, object, object)
    

    def __init__(self, fft_pol0, fft_pol1, channels_pol0, channels_pol1, outliers_indices_pol0, outliers_indices_pol1, parent=None):
        super().__init__(parent)

        self.fft_pol0 = fft_pol0
        self.fft_pol1 = fft_pol1
        self.channels_pol0 = channels_pol0
        self.channels_pol1 = channels_pol1
        self.outliers_indices_pol0 = outliers_indices_pol0
        self.outliers_indices_pol1 = outliers_indices_pol1


    def clean_fft_statistical(self, fft_data, outliers):
        fft_clean = fft_data.copy()
        n_iter, nchan = fft_clean.shape

        for ch in outliers:
            # prendi distribuzione del canale (tutte le iterazioni)
            channel_values = fft_clean[:, ch]

            # stima robusta (mediana complessa)
            median_real = np.median(channel_values.real)
            median_imag = np.median(channel_values.imag)

            robust_value = median_real + 1j * median_imag

            fft_clean[:, ch] = robust_value

        return fft_clean


    
    def clean_fft(self, fft_data, outliers):

        fft_clean = fft_data.copy()
        n_iter, nchan = fft_clean.shape

        for ch in outliers:

            left = max(ch - 2, 0)
            right = min(ch + 3, nchan)

            neighbors = list(range(left, ch)) + list(range(ch+1, right))

            if len(neighbors) == 0:
                continue

            # media complessa ? preserva fase
            fft_clean[:, ch] = np.mean(fft_clean[:, neighbors], axis=1)

        return fft_clean
        
   
    
    def run(self):



        self.update_statusbar_signal.emit('Reconstructing the FFT coefficients...')   
        self.update_button_signal.emit(False)  
        self.update_progressbar_range.emit(0, 0)
        
        # Clean FFT using outliers indices
        clean_fft_pol0 = self.clean_fft_statistical(self.fft_pol0, self.outliers_indices_pol0)
        clean_fft_pol1 = self.clean_fft_statistical(self.fft_pol1, self.outliers_indices_pol1)

        # IFFT ? reconstruct time domain
        ifft_pol0 = np.fft.ifft(clean_fft_pol0, axis=1)
        ifft_pol1 = np.fft.ifft(clean_fft_pol1, axis=1)

        # opzionale: rimuovi DC
        ifft_pol0 -= np.mean(ifft_pol0)
        ifft_pol1 -= np.mean(ifft_pol1)

        # PSD ricostruito (verifica finale)
        psd_pol0 = np.abs(clean_fft_pol0) ** 2
        psd_pol1 = np.abs(clean_fft_pol1) ** 2

        psd_tot_pol0 = psd_pol0.sum(axis=0)
        psd_tot_pol1 = psd_pol1.sum(axis=0)

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(100)   


        self.result_signal_psd.emit(
            psd_tot_pol0,
            psd_tot_pol1,
            clean_fft_pol0,
            clean_fft_pol1
        )


        self.update_button_signal.emit(True)  

        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100) 
        
        #return self.pol0_psd_tot, self.pol1_psd_tot






class CreateFilteredSpectrum(QThread):

    # Signal to notify the main thread to disable/enable the button in the interface
    update_button_signal = pyqtSignal(bool)
    # Signal to notify the main thread (UI) to update the status bar in the interface 
    update_statusbar_signal = pyqtSignal(str)
    # Signal to notify the main thread (UI) to update the progressbar current value
    update_progressbar_value = pyqtSignal(int)
     # Signal to notify the main thread (UI) to update the progressbar current range
    update_progressbar_range = pyqtSignal(int, int)
    # Signal to notify the main thread with the result of the processing
    result_signal_psd = pyqtSignal(object, object, object, object)
    

    def __init__(self, fft_pol0, fft_pol1, channels_pol0, channels_pol1, outliers_indices_pol0, outliers_indices_pol1, parent=None):
        super().__init__(parent)

        self.fft_pol0 = fft_pol0
        self.fft_pol1 = fft_pol1
        self.channels_pol0 = channels_pol0
        self.channels_pol1 = channels_pol1
        self.outliers_indices_pol0 = outliers_indices_pol0
        self.outliers_indices_pol1 = outliers_indices_pol1
        
    def run(self):


        self.update_statusbar_signal.emit('Reconstructing the FFT coefficients...')   
        self.update_button_signal.emit(False)  
        self.update_progressbar_range.emit(0, 0)


        '''
        
        # Process each element individually
        for i in range(len(self.channels_pol0)):

            if(self.channels_pol0[i] < 1018):

                while True:
                    # Add 5 to the current element
                    self.channels_pol0[i] += 1 
                                
                    # Check if the updated element differs by at least 5 from all other elements in the original array
                    all_differences_ok = True
                    for j in range(len(self.outliers_indices_pol0)):
                        #if i != j:  # Compare with all other elements
                        difference = abs(self.channels_pol0[i] - self.outliers_indices_pol0[j])
                        if difference < 5:
                            all_differences_ok = False
                            break
                    
                    # If the differences are all 5 or greater, exit the loop for this element
                    if all_differences_ok:
                        break

        print(self.channels_pol0)

        # Process each element individually
        for i in range(len(self.channels_pol1)):

            if(self.channels_pol1[i] < 1018):
                
                while True:
                    # Add 5 to the current element
                    self.channels_pol1[i] += 1 
                
                    
                    # Check if the updated element differs by at least 5 from all other elements in the original array
                    all_differences_ok = True
                    for j in range(len(self.outliers_indices_pol1)):
                        #if i != j:  # Compare with all other elements
                        difference = abs(self.channels_pol1[i] - self.outliers_indices_pol1[j])
                        if difference < 5:
                            all_differences_ok = False
                            break
                    
                    # If the differences are all 5 or greater, exit the loop for this element
                    if all_differences_ok:
                        break

        print(self.channels_pol1)
        '''

        # --- RICERCA CANALE SANO POL 0 ---
        new_channels0 = []
        for out_idx in self.outliers_indices_pol0:
            found = False
            for dist in range(5, 512): 
                for sign in [1, -1]:
                    candidate = out_idx + (dist * sign)
                    if 0 <= candidate < 1024:
                        is_safe = True
                        for other_out in self.outliers_indices_pol0:
                            if abs(candidate - other_out) < 5:
                                is_safe = False
                                break
                        if is_safe:
                            new_channels0.append(candidate)
                            found = True
                            break
                if found: break
            if not found: new_channels0.append((out_idx + 10) % 1024)

        self.channels_pol0 = new_channels0

        # --- RICERCA CANALE SANO POL 1 ---
        new_channels1 = []
        for out_idx in self.outliers_indices_pol1:
            found = False
            for dist in range(5, 512):
                for sign in [1, -1]:
                    candidate = out_idx + (dist * sign)
                    if 0 <= candidate < 1024:
                        is_safe = True
                        for other_out in self.outliers_indices_pol1:
                            if abs(candidate - other_out) < 5:
                                is_safe = False
                                break
                        if is_safe:
                            new_channels1.append(candidate)
                            found = True
                            break
                if found: break
            if not found: new_channels1.append((out_idx + 10) % 1024)

        self.channels_pol1 = new_channels1

        # --- STAMPA DI VERIFICA (DEBUG) ---
        print("\n--- Mapping Outliers -> Safe Channels (Pol 0) ---")
        for i in range(len(self.outliers_indices_pol0)):
            print(f"Outlier: {self.outliers_indices_pol0[i]:4d} --> Safe: {self.channels_pol0[i]:4d} (Dist: {self.channels_pol0[i] - self.outliers_indices_pol0[i]:+d})")

        print("\n--- Mapping Outliers -> Safe Channels (Pol 1) ---")
        for i in range(len(self.outliers_indices_pol1)):
            print(f"Outlier: {self.outliers_indices_pol1[i]:4d} --> Safe: {self.channels_pol1[i]:4d} (Dist: {self.channels_pol1[i] - self.outliers_indices_pol1[i]:+d})")
        print("--------------------------------------------------\n")

        # --- PRE-CALCOLO MAGNITUDO E FASE ---
        # Calcoliamo tutto una volta sola per evitare di ripetere np.abs/np.angle
        mag0 = np.abs(self.fft_pol0)
        phase0 = np.angle(self.fft_pol0)
        mag1 = np.abs(self.fft_pol1)
        phase1 = np.angle(self.fft_pol1)

        # Creiamo le matrici pulite partendo dalle originali
        clean_mag0 = mag0.copy()
        clean_mag1 = mag1.copy()

        num_iterations = self.fft_pol0.shape[0] # tipicamente 100

        # --- RIPARAZIONE POL0 ---
        # Sfruttiamo il fatto che i canali sani sono gi� stati validati
        for out_idx, safe_idx in zip(self.outliers_indices_pol0, self.channels_pol0):
            # Statistiche calcolate sulla colonna (asse del tempo/iterazioni)
            safe_col = mag0[:, safe_idx]
            m, s = np.mean(safe_col), np.std(safe_col)
            
            # Generazione veloce dell'intera colonna di rumore gaussiano
            # np.maximum(..., 0) assicura che la magnitudo non sia mai negativa
            clean_mag0[:, out_idx] = np.maximum(np.random.normal(m, s, size=num_iterations), 0)

        # --- RIPARAZIONE POL1 ---
        for out_idx, safe_idx in zip(self.outliers_indices_pol1, self.channels_pol1):
            safe_col = mag1[:, safe_idx]
            m, s = np.mean(safe_col), np.std(safe_col)
            clean_mag1[:, out_idx] = np.maximum(np.random.normal(m, s, size=num_iterations), 0)

        # --- RICOSTRUZIONE COEFFICIENTI FFT E PSD ---
        # Operazioni matriciali super veloci
        self.update_statusbar_signal.emit('Squaring and integrating...')
        
        # Ricostruiamo i complessi (Z = Mag * e^(j*Phase))
        clean_fft_pol0 = clean_mag0 * np.exp(1j * phase0)
        clean_fft_pol1 = clean_mag1 * np.exp(1j * phase1)

       
        # PSD Media: Calcoliamo lo spettro di potenza medio (magnitudo al quadrato)
        # Usiamo np.mean per mantenere l'ampiezza coerente con i dati originali,
        # evitando che il valore scali con il numero di iterazioni.
        # Pi� veloce di np.abs(clean_fft)**2 perch� usiamo la magnitudo gi� calcolata.
        psd0_tot = np.mean(clean_mag0**2, axis=0)
        psd1_tot = np.mean(clean_mag1**2, axis=0)

    

        # --- PULIZIA DC (OPZIONALE MA CONSIGLIATA) ---
        # psd0_tot[0] = psd0_tot[1]
        # psd1_tot[0] = psd1_tot[1]

        self.result_signal_psd.emit(psd0_tot, psd1_tot, clean_fft_pol0, clean_fft_pol1)

        self.update_button_signal.emit(True)  

        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100) 
     



        '''
        magnitude_pol0 = np.abs(self.fft_pol0)
        phase_pol0 = np.angle(self.fft_pol0) # np.angle(z) == np.arctan2(b, a)

        magnitude_pol1 = np.abs(self.fft_pol1)
        phase_pol1 = np.angle(self.fft_pol1) # np.angle(z) == np.arctan2(b, a)

        #print('Initial FFT coeff.:', self.fft_pol0[0][185:195])
        #print('Magnitude [0]:', magnitude[0][185:195])
        #print('Magnitude [1]:', magnitude[1][185:195])
        #print('Phase', phase[0][185:195])

        # Step 2: Smooth the power spectrum using a median filter
        #smooth_power = medfilt(self.psd_pol0, kernel_size=5)
        #smooth_magnitude = np.sqrt(smooth_power)

        #print(smooth_magnitude)

        # Step 3: Create a clean magnitude array (initially a copy of original)
        clean_magnitude_pol0 = magnitude_pol0.copy()
        clean_magnitude_pol1 = magnitude_pol1.copy()
        #print('Clean Magnitude copy [0]:', clean_magnitude[0][185:195])
        #print('Clean Magnitude copy [1]:', clean_magnitude[1][185:195])


        for i in range(len(self.channels_pol0)):

            channel_values_pol0 = magnitude_pol0[:, self.channels_pol0[i]]
            #print('column_values', channel_values_pol0)

            # Compute average (mean)
            mean_val = np.mean(channel_values_pol0)

            # Compute variance
            std_dev = np.std(channel_values_pol0)

            print(f"Mean: {mean_val}")
            print(f"Variance: {std_dev}")

            # Compute the interval
            lower_bound = mean_val - std_dev
            upper_bound = mean_val + std_dev
            #lower_bound = min(channel_values_pol0)
            #upper_bound = max(channel_values_pol0)

            # Get a random value within the interval
            random_value = np.random.uniform(lower_bound, upper_bound)

            print(f"Interval: ({lower_bound}, {upper_bound})")
            print(f"Random value from the interval: {random_value}")

            # Replace all values in columns related to the outliers with random values from the interval
            #for i in range(len(outliers)):
            #clean_magnitude_pol0[:, self.outliers_indices_pol0[i]] = np.random.uniform(lower_bound, upper_bound, size=clean_magnitude_pol0.shape[0])*1.09
            clean_magnitude_pol0[:, self.outliers_indices_pol0[i]] = np.random.normal(mean_val, std_dev, size=clean_magnitude_pol0.shape[0]) # we use a Gauss distribution
         



        # Step 4: Replace outlier magnitudes using the smoothed version
        #clean_magnitude[0][self.outliers_indices_pol0] = smooth_magnitude[self.outliers_indices_pol0]
        #print('Clean Magnitude copy replaced [0]:', clean_magnitude[0][185:195])
        #print('Clean Magnitude copy replaced [1]:', clean_magnitude[1][185:195])

        # Step 5: Reconstruct the cleaned FFT coefficients
        clean_fft_coeffs_pol0 = clean_magnitude_pol0 * np.exp(1j * phase_pol0)
       

        #print('Clean FFT coeff [0]', clean_fft_coeffs[0][185:195])
        #print('Clean FFT coeff [1]', clean_fft_coeffs[1][185:195])


        for i in range(len(self.channels_pol1)):

            channel_values_pol1 = magnitude_pol1[:, self.channels_pol1[i]]
            #print('column_values', channel_values_pol1)

            # Compute average (mean)
            mean_val = np.mean(channel_values_pol1)

            # Compute variance
            std_dev = np.std(channel_values_pol1)

            print(f"Mean: {mean_val}")
            print(f"Variance: {std_dev}")

            # Compute the interval
            lower_bound = mean_val - std_dev
            upper_bound = mean_val + std_dev

            # Get a random value within the interval
            random_value = np.random.uniform(lower_bound, upper_bound)

            print(f"Interval: ({lower_bound}, {upper_bound})")
            print(f"Random value from the interval: {random_value}")

            # Replace all values in columns related to the outliers with random values from the interval
            #for i in range(len(outliers)):
            #clean_magnitude_pol1[:, self.outliers_indices_pol1[i]] = np.random.uniform(lower_bound, upper_bound, size=clean_magnitude_pol1.shape[0])*1.09
            clean_magnitude_pol1[:, self.outliers_indices_pol1[i]] = np.random.normal(mean_val, std_dev, size=clean_magnitude_pol1.shape[0]) # we use a Gauss distribution


        # Step 4: Replace outlier magnitudes using the smoothed version
        #clean_magnitude[0][self.outliers_indices_pol0] = smooth_magnitude[self.outliers_indices_pol0]
        #print('Clean Magnitude copy replaced [0]:', clean_magnitude[0][185:195])
        #print('Clean Magnitude copy replaced [1]:', clean_magnitude[1][185:195])

        # Step 5: Reconstruct the cleaned FFT coefficients
        clean_fft_coeffs_pol1 = clean_magnitude_pol1 * np.exp(1j * phase_pol1)

        #print('Clean FFT coeff [0]', clean_fft_coeffs[0][185:195])
        #print('Clean FFT coeff [1]', clean_fft_coeffs[1][185:195])



        #xxx = np.fft.ifft(clean_fft_coeffs_pol0)
        #print(max(xxx[0]))

        self.update_progressbar_value.emit(100) 

        self.update_statusbar_signal.emit('Squaring the FFT coefficients...')   
        self.update_button_signal.emit(False)   
        self.update_progressbar_value.emit(0)   

        
        # Calculate the power spectrum (magnitude squared) for all iterations
        power_spectrum_pol0_corrected = np.array(np.abs(clean_fft_coeffs_pol0) ** 2)
        power_spectrum_pol1_corrected = np.array(np.abs(clean_fft_coeffs_pol1) ** 2)
       
        # Sum the elements with the same index in each row
        #self.pol0_psd_tot_corrected = [sum(x) for x in zip(*power_spectrum_pol0_corrected)]
        #self.pol1_psd_tot_corrected = [sum(x) for x in zip(*power_spectrum_pol1_corrected)]
        self.pol0_psd_tot_corrected = power_spectrum_pol0_corrected.sum(axis=0)
        self.pol1_psd_tot_corrected = power_spectrum_pol1_corrected.sum(axis=0)

        #self.pol0_psd_tot_corrected[0] = self.pol0_psd_tot_corrected[1] # it removes the spike at 0 for better plotting
        #self.pol1_psd_tot_corrected[0] = self.pol1_psd_tot_corrected[1] # it removes the spike at 0 for better plotting

        self.update_statusbar_signal.emit('Power Spectrum ready.')
        self.update_progressbar_value.emit(100)   


        self.result_signal_psd.emit(self.pol0_psd_tot_corrected, self.pol1_psd_tot_corrected,  clean_fft_coeffs_pol0,  clean_fft_coeffs_pol1)

        self.update_button_signal.emit(True)  

        self.update_progressbar_range.emit(0, 100)
        self.update_progressbar_value.emit(100) 
        
        #return self.pol0_psd_tot, self.pol1_psd_tot
        '''



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

    '''
    def _find_outliers(self, spectrum, chunk_size, threshold):

        outlier_indices = []
        n = len(spectrum)
        num_chunks = n // chunk_size

        for i in range(num_chunks):

            start = i * chunk_size
            end = start + chunk_size
            chunk = spectrum[start:end]

            mean = np.mean(chunk)
            std = np.std(chunk)

            if std == 0:
                continue

            z_scores = (chunk - mean) / std
            mask = np.abs(z_scores) > threshold

            indices = np.where(mask)[0] + start
            outlier_indices.extend(indices.tolist())

        # remainder chunk
        remainder = n % chunk_size
        if remainder > 0:
            start = num_chunks * chunk_size
            chunk = spectrum[start:]

            mean = np.mean(chunk)
            std = np.std(chunk)

            if std != 0:
                z_scores = (chunk - mean) / std
                mask = np.abs(z_scores) > threshold

                indices = np.where(mask)[0] + start
                outlier_indices.extend(indices.tolist())

        return outlier_indices
    '''
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







class DetectOutliers_old(QThread):

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


    def __init__(self, power_spectrum_pol0, chunk_size_pol0, threshold_pol0, power_spectrum_pol1, chunk_size_pol1, threshold_pol1,parent=None):
        super().__init__(parent)

        self.power_spectrum_pol0 = power_spectrum_pol0
        self.chunk_size_pol0 = chunk_size_pol0
        self.threshold_pol0 = threshold_pol0

        self.power_spectrum_pol1 = power_spectrum_pol1
        self.chunk_size_pol1 = chunk_size_pol1
        self.threshold_pol1 = threshold_pol1

    def run(self):

        self.update_statusbar_signal.emit('Inspecting the Power Spectrum for outliers...')
        self.update_button_signal.emit(False) 
        self.update_progressbar_range.emit(0,0)
        time.sleep(1)

        np.set_printoptions(legacy='1.25') # in case a print is needed it avouds formatting like np.float64(.....) for each point

        outlier_indices_pol0 = []
        num_chunks_pol0 = len(self.power_spectrum_pol0) // self.chunk_size_pol0  # Number of full chunks

        for i in range(num_chunks_pol0):
            # Step 2: Define the chunk
            start = i * self.chunk_size_pol0
            end = (i + 1) * self.chunk_size_pol0
            chunk = self.power_spectrum_pol0[start:end]

            # Step 3: Calculate mean and standard deviation for the chunk
            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)

            # Step 4: Calculate Z-scores for each value in the chunk
            z_scores = (chunk - chunk_mean) / chunk_std

            # Step 5: Identify outliers in this chunk
            outliers = np.where(np.abs(z_scores) > self.threshold_pol0)
            outlier_indices_pol0.extend(start + outliers[0])  # Adjust the indices to the global spectrum


        # Handle the last chunk (if any remaining bins)
        remainder = len(self.power_spectrum_pol0) % self.chunk_size_pol0
        if remainder > 0:
            start = num_chunks_pol0 * self.chunk_size_pol0
            end = len(self.power_spectrum_pol0)
            chunk = self.power_spectrum_pol0[start:end]

            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)

            z_scores = (chunk - chunk_mean) / chunk_std
            outliers = np.where(np.abs(z_scores) > self.threshold_pol0)
            outlier_indices_pol0.extend(start + outliers[0])

        # Step 8: Output the indices and values of the outliers
        print("Indices of outliers:", outlier_indices_pol0)
        #print("Values of outliers:", spectrum[outlier_indices])
        #self.outliers_signal_psd.emit(outlier_indices_pol0)
        #return outlier_indices

        outlier_indices_pol1 = []
        num_chunks_pol1 = len(self.power_spectrum_pol1) // self.chunk_size_pol1  # Number of full chunks

        for i in range(num_chunks_pol1):
            # Step 2: Define the chunk
            start = i * self.chunk_size_pol1
            end = (i + 1) * self.chunk_size_pol1
            chunk = self.power_spectrum_pol1[start:end]

            # Step 3: Calculate mean and standard deviation for the chunk
            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)

            # Step 4: Calculate Z-scores for each value in the chunk
            z_scores = (chunk - chunk_mean) / chunk_std

            # Step 5: Identify outliers in this chunk
            outliers = np.where(np.abs(z_scores) > self.threshold_pol1)
            outlier_indices_pol1.extend(start + outliers[0])  # Adjust the indices to the global spectrum


        # Handle the last chunk (if any remaining bins)
        remainder = len(self.power_spectrum_pol1) % self.chunk_size_pol1
        if remainder > 0:
            start = num_chunks_pol0 * self.chunk_size_pol1
            end = len(self.power_spectrum_pol1)
            chunk = self.power_spectrum_pol1[start:end]

            chunk_mean = np.mean(chunk)
            chunk_std = np.std(chunk)

            z_scores = (chunk - chunk_mean) / chunk_std
            outliers = np.where(np.abs(z_scores) > self.threshold_pol1)
            outlier_indices_pol0.extend(start + outliers[0])

        # Step 8: Output the indices and values of the outliers
        print("Indices of outliers:", outlier_indices_pol1)
        #print("Values of outliers:", spectrum[outlier_indices])
        self.outliers_signal_psd.emit(outlier_indices_pol0, outlier_indices_pol1)
        #return outlier_indices



class CreateFilteredTimeDomainSignal(QThread):
    update_button_signal = pyqtSignal(bool)
    update_statusbar_signal = pyqtSignal(str)
    update_progressbar_value = pyqtSignal(int)
    result_signal_filtered_time_domain = pyqtSignal(object, object, object, object)

    def __init__(self, fft_corrected_pol0, fft_corrected_pol1, outliers_index_pol0, outliers_index_pol1, parent=None):
        super().__init__(parent)
        # Usiamo i coefficienti FFT gi� riparati dalla classe precedente
        self.fft_corrected_pol0 = fft_corrected_pol0
        self.fft_corrected_pol1 = fft_corrected_pol1
        self.outliers_index_pol0 = outliers_index_pol0
        self.outliers_index_pol1 = outliers_index_pol1

    def run(self):
        self.update_button_signal.emit(False)
        self.update_statusbar_signal.emit('Ricostruzione del segnale tempo-variante (8-bit uint)...')

        # 1. Eseguiamo la IFFT (Trasformata Inversa veloce di Fourier)
        # NumPy la esegue riga per riga (axis=1) automaticamente se la matrice � 2D
        ifft0 = np.fft.ifft(self.fft_corrected_pol0, axis=1)
        ifft1 = np.fft.ifft(self.fft_corrected_pol1, axis=1)

        # 2. Rimozione dell'offset DC
        # Sottraiamo la media per assicurarci che il segnale sia centrato perfettamente sullo zero float
        ifft0 -= np.mean(ifft0, axis=1, keepdims=True)
        ifft1 -= np.mean(ifft1, axis=1, keepdims=True)

        # 3. Separazione Componenti Reale e Immaginaria
        # In radioastronomia (segnali IQ), la parte reale e immaginaria sono i canali in fase e quadratura
        real0, imag0 = np.real(ifft0), np.imag(ifft0)
        real1, imag1 = np.real(ifft1), np.imag(ifft1)

        # 4. Scaling, Clipping e Conversione in uint8
        # Spieghiamo il processo:
        # - Aggiungiamo 128 perch� lo zero fisico dei dati uint8 � a met� scala (128).
        # - np.clip assicura che eventuali oscillazioni non vadano oltre [0, 255] (evita l'overflow).
        # - np.round gestisce l'arrotondamento corretto prima del passaggio a intero.
        
        def to_uint8(data):
            return np.clip(np.round(data + 128), 0, 255).astype(np.uint8)

        # Applichiamo la conversione a tutte le componenti
        r_uint8_0 = to_uint8(real0)
        i_uint8_0 = to_uint8(imag0)
        r_uint8_1 = to_uint8(real1)
        i_uint8_1 = to_uint8(imag1)

        # 5. Invio dei dati pronti per essere salvati nel file binario originale
        self.result_signal_filtered_time_domain.emit(r_uint8_0, i_uint8_0, r_uint8_1, i_uint8_1)

        self.update_statusbar_signal.emit('Segnale tempo-dominio ricostruito con successo.')
        self.update_button_signal.emit(True)
        self.update_progressbar_value.emit(100)





class CreateFilteredTimeDomainSignal_old(QThread): # this class performs an ifft to the FFT coefficients and goes back to the 8-bit unsigned integers

    # Signal to notify the main thread to disable/enable the button in the interface
    update_button_signal = pyqtSignal(bool)
    # Signal to notify the main thread (UI) to update the status bar in the interface 
    update_statusbar_signal = pyqtSignal(str)
    # Signal to notify the main thread (UI) to update the progressbar current value
    update_progressbar_value = pyqtSignal(int)
    # Signal to notify the main thread with the result of the processing
    result_signal_filtered_time_domain = pyqtSignal(object, object, object, object)
    


    def __init__(self, fft_corrected_pol0, fft_corrected_pol1, outliers_index_pol0, outliers_index_pol1, parent=None):
        super().__init__(parent)

        self.fft_corrected_pol0 = fft_corrected_pol0
        self.fft_corrected_pol1 = fft_corrected_pol1

        self.outliers_index_pol0 = outliers_index_pol0
        self.outliers_index_pol1 = outliers_index_pol1
        
    def run(self):


        self.update_statusbar_signal.emit('Reconstructiong the time domain signal as 8-bit unsigned integers...')   
        self.update_button_signal.emit(False)   

        '''
        for i in range(len(self.outliers_index_pol0)):

          
            print(self.fft_corrected_pol0[0,[self.outliers_index_pol0[i]]])

            self.fft_corrected_pol0[:,[self.outliers_index_pol0[i]]] = np.sqrt((abs(self.fft_corrected_pol0[:,[self.outliers_index_pol0[i]+5]])**2 + 
                (abs(self.fft_corrected_pol0[:,[self.outliers_index_pol0[i]-5]]))**2) / 2) * np.exp(1j * np.angle(self.fft_corrected_pol0[:,[self.outliers_index_pol0[i]]]))
            print(self.fft_corrected_pol0[0,[self.outliers_index_pol0[i]]])'''

        #self.fft_corrected_pol0[0] = 0 # Remove DC offset generated by ifft at channel 0
        #self.fft_corrected_pol1[0] = 0 # Remove DC offset generated by ifft at channel 0

        ifft_pol0 = np.fft.ifft(self.fft_corrected_pol0)
        ifft_pol1 = np.fft.ifft(self.fft_corrected_pol1)

        # Step 6: Remove DC offset (mean) from the time-domain result
        ifft_pol0 -= np.mean(ifft_pol0)
        ifft_pol1 -= np.mean(ifft_pol1)

        #print('len ifft_pol0', len(ifft_pol0))
        #ifft_pol0[0] = 0
        #ifft_pol1[0] = 0

        #print(recovered_time_domain_signal_pol0)
        #print(recovered_time_domain_signal_pol1)
        #print(len(recovered_time_domain_signal_pol0))
        #print(len(recovered_time_domain_signal_pol1))

        #print(min(recovered_time_domain_signal_pol0[0]))
        #print(max(recovered_time_domain_signal_pol0[1]))

        real_part_pol0 = np.real(ifft_pol0)
        imag_part_pol0 = np.imag(ifft_pol0)
        real_part_pol1 = np.real(ifft_pol1)
        imag_part_pol1 = np.imag(ifft_pol1)

        #for i in range(len(self.outliers_index_pol0)):

        #    real_part_pol0[:,[self.outliers_index_pol0[i]]] = real_part_pol0[:,[self.outliers_index_pol0[i]]] * 1.15       #

        #for i in range(len(self.outliers_index_pol1)):

        #    real_part_pol1[:,[self.outliers_index_pol1[i]]] = real_part_pol1[:,[self.outliers_index_pol1[i]]] * 1.15        
 
        
        
        

       

        # Since float numbers are in a range largen than [0, 255] we clip the values outside the range
        # This anyway should not introduce artefacts in the real data 
        real_clipped_pol0 = np.clip(real_part_pol0 +128, 0, 255)
        imag_clipped_pol0 = np.clip(imag_part_pol0 +128, 0, 255)
        real_clipped_pol1 = np.clip(real_part_pol1 +128, 0, 255)
        imag_clipped_pol1 = np.clip(imag_part_pol1 +128, 0, 255)
        # go back to unsigned 8-bit integers as the original data
        real_uint8_pol0 = np.round(real_clipped_pol0).astype(np.uint8)
        imag_uint8_pol0 = np.round(imag_clipped_pol0).astype(np.uint8)
        real_uint8_pol1 = np.round(real_clipped_pol1).astype(np.uint8)
        imag_uint8_pol1 = np.round(imag_clipped_pol1).astype(np.uint8)

        #reconstructed_uint8_pol0 = np.stack([real_uint8_pol0, imag_uint8_pol0], axis=1)
        #reconstructed_uint8_pol1 = np.stack([real_uint8_pol1, imag_uint8_pol1], axis=1)

        self.result_signal_filtered_time_domain.emit(real_uint8_pol0, imag_uint8_pol0, real_uint8_pol1, imag_uint8_pol1)
    
    
    
    
    
    
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