import os
import numpy as np
from PyQt5.uic.uiparser import QtWidgets

from gui.src.dada_analyzer import CreateFilteredSpectrum, CreateRawSpectrum, CreateFilteredTimeDomainSignal
from gui.src.dada_analyzer import DetectOutliers
from gui.src.graphcreator import GraphCreator
from gui.src.dada_file_manager import dada_file_manager

from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog

#from scipy.signal import medfilt



class PowerSpectrumAnalyzer():

    WPOINTS = ['2', '4', '8', '16', '32', '64', '128', '256', '512', '1024']
    PROCESSING_METHODS = ['Z-scores']
    MAX_SLIDER_VAL = 7

    selected_data_file = ""
    # Some parameters for the data processing
    f0 = 0 # Cental Frequency in Mhz
    bw = 0 # Bandwidth in MHz
    tsamp = 0 # sampling time in us, then converted to seconds
    file_size = 0 # file size in bytes
    # Additional data (at the moment hard-coded)
    points_per_iteration = 1024 # Number of points to be read out per iteration
    number_of_channels = 1024 # Number of channels in the Power Spectrum
    iterations = 0 # Number of iterations selected by the User
    max_iterations = 0 # Number of possible iterations according to the .dada file length

    data = None # variable containing the whole dataset reshaped (n raws x 4 columns) which is loaded with the button 'Load'
    dada_file_object = None
    
    psd_pol0 = None # the raw power spectrum density for pol0
    psd_pol1 = None # the raw power spectrum density for pol1
    fft_pol0 = None # the raw fft coefficients for pol0
    fft_pol1 = None # the raw fft coefficients for pol1
    outliers_indices_pol0 = None
    outliers_indices_pol1 = None

    fft_corrected_pol0 = None
    fft_corrected_pol1 = None

    destination_path = ""
    

    def __init__(self, load_btn, save_btn, start_btn, filter_btn, reset_btn, wpoints_pol_left_cmb, wpoints_pol_right_cmb,
            threshold_pol_left_psa_hs, threshold_pol_right_psa_hs, iterations_le, data_processing_method_psa_cb,
            data_processing_start_psa_pbar, data_processing_filter_psa_pbar, threshold_pol_left_lb, threshold_pol_right_lb, iterations_lb,
            gp_vl_raw_data, gp_vl_filtered_data, source_name_val, ra_val, dec_val,
            telescope_val, instrument_val, frequency_val, bandwidth_val, sampling_time_val, iterations_val,
            interations_per_second_val, points_per_iteration_val, spectrum_channels_val, filename_val, statusbar):

        #self.debug_on = debug_on

        self.load_btn = load_btn
        self.save_btn = save_btn
        self.start_btn = start_btn
        self.filter_btn = filter_btn
        self.reset_btn = reset_btn
        self.wpoints_pol_left_cmb = wpoints_pol_left_cmb
        self.wpoints_pol_right_cmb = wpoints_pol_right_cmb
        self.threshold_pol_left_psa_hs = threshold_pol_left_psa_hs
        self.threshold_pol_right_psa_hs = threshold_pol_right_psa_hs
        self.iterations_le = iterations_le
        self.data_processing_method_psa_cb = data_processing_method_psa_cb
        self.data_processing_start_psa_pbar = data_processing_start_psa_pbar
        self.data_processing_filter_psa_pbar = data_processing_filter_psa_pbar    
        self.threshold_pol_left_lb = threshold_pol_left_lb
        self.threshold_pol_right_lb = threshold_pol_right_lb
        self.iterations_lb = iterations_lb
        self.gp_vl_raw_data = gp_vl_raw_data
        self.gp_vl_filtered_data = gp_vl_filtered_data
        self.source_name_val = source_name_val
        self.ra_val = ra_val
        self.dec_val = dec_val
        self.telescope_val = telescope_val
        self.instrument_val = instrument_val
        self.frequency_val = frequency_val
        self.bandwidth_val = bandwidth_val
        self.sampling_time_val = sampling_time_val
        self.iterations_val = iterations_val
        self.interations_per_second_val = interations_per_second_val
        self.points_per_iteration_val = points_per_iteration_val
        self.spectrum_channels_val = spectrum_channels_val
        self.filename_val = filename_val
        self.statusbar = statusbar

        # Itialize progress bar values
        self.data_processing_start_psa_pbar.setValue(0)
        self.data_processing_filter_psa_pbar.setValue(0)

        # Add combobox drop down list values
        self.data_processing_method_psa_cb.addItems(self.PROCESSING_METHODS)
        self.wpoints_pol_left_cmb.addItems(self.WPOINTS)
        self.wpoints_pol_right_cmb.addItems(self.WPOINTS)
       
        # Set the max number of visible items in the combobox dropdown list
        self.data_processing_method_psa_cb.setMaxVisibleItems(5) 
        self.wpoints_pol_left_cmb.setMaxVisibleItems(5) 
        self.wpoints_pol_right_cmb.setMaxVisibleItems(5)

        # Set visible item of the combobox dropdown list 
        self.wpoints_pol_left_cmb.setCurrentIndex(4)
        self.wpoints_pol_right_cmb.setCurrentIndex(4)
       
        # Set the lower and upper limits of the threshold for the horizontal sliders
        self.threshold_pol_left_psa_hs.setMinimum(0)  # Lower limit 
        self.threshold_pol_left_psa_hs.setMaximum(100)
        # Set tick position (above the slider in this case)
        #self.threshold_pol_left_psa_hs.setTickPosition(QSlider.TicksBelow)
        # Set tick interval (ticks appear every 10 units)
        self.threshold_pol_left_psa_hs.setTickInterval(100)
        self.threshold_pol_left_psa_hs.setValue(int(round(3*100/self.MAX_SLIDER_VAL))) # 7 corresponds to 100
        self.threshold_pol_right_psa_hs.setMinimum(0)  # Lower limit 
        self.threshold_pol_right_psa_hs.setMaximum(100) 
        self.threshold_pol_right_psa_hs.setTickInterval(100)
        self.threshold_pol_right_psa_hs.setValue(int(round(3*100/self.MAX_SLIDER_VAL))) # 7 corresponds to 100
        # Set Slider listeners
        self.threshold_pol_left_psa_hs.valueChanged.connect(self.slider_changed_value_pol_left)
        self.threshold_pol_right_psa_hs.valueChanged.connect(self.slider_changed_value_pol_right)
        
        # Buttons settings
        self.reset_buttons()
        
        # Buttons actions
        self.load_btn.clicked.connect(self.open_file_dialog)  # Connect the button click event to open the file dialog
        self.save_btn.clicked.connect(self.save_file_dialog)
        self.start_btn.clicked.connect(self.power_spectrum_builder)
        self.filter_btn.clicked.connect(self.filter_data)
        self.reset_btn.clicked.connect(self.reset_data)

        self.iterations_le.setEnabled(False)
        # Connect the text change signal to check the value
        self.iterations_le.textChanged.connect(self.check_iterations_input)

        # Set threshold label text
        self.threshold_pol_left_lb.setText('THRESHOLD [3.0]')
        self.threshold_pol_right_lb.setText('THRESHOLD [3.0]')
        # Set iteration label text
        self.iterations_lb.setText('ITERATIONS [0]')

        # Add an empty plot in the plot area
        self.graph = GraphCreator()
        canvas_raw_data = self.graph.create_MPL_empty_plot('RAW DATA')
        canvas_filtered_data = self.graph.create_MPL_empty_plot('FILTERED DATA')
        self.gp_vl_raw_data.addWidget(canvas_raw_data)
        self.gp_vl_filtered_data.addWidget(canvas_filtered_data)

        
    
    def check_iterations_input(self):
        # Get the current input value from the QLineEdit
        input_value = self.iterations_le.text()

        # Check if the input value exceeds the maximum allowed (55031)
        if input_value.isdigit() and int(input_value) > self.max_iterations:
            self.iterations_le.setText(str(self.max_iterations))  # Reset the value to 55031 if it exceeds
            #print("Exceeded maximum value. Reset to 55031.")

        # Enable 'START' button if the number of iterations given by the user is > 0
        if(input_value != ''):
            if(int(input_value) > 0): 
                self.start_btn.setEnabled(True)
            else:
                self.start_btn.setEnabled(False)
        else:
            self.start_btn.setEnabled(False)


    
    def disable_buttons(self, value):

        self.load_btn.setEnabled(value)
        self.save_btn.setEnabled(value)
        self.start_btn.setEnabled(value)
        self.filter_btn.setEnabled(value)
        self.reset_btn.setEnabled(value)



    def display_plot(self, psd_pol0, psd_pol1, fft_pol0, fft_pol1):

        # Store the raw power spectra
        self.psd_pol0 = psd_pol0
        self.psd_pol1 = psd_pol1

        # Store the fft coefficients for pol0 and pol1
        self.fft_pol0 = fft_pol0
        self.fft_pol1 = fft_pol1


        np.set_printoptions(legacy='1.25') # in case a print is needed it avouds formatting like np.float64(.....) for each point

        # Check if the vertical layout has already a widget (canvas and toolbar)
        # Optionally, delete the widget to free up resources
        # self.button2.deleteLater()
        if(self.gp_vl_raw_data.count() > 0):

            for i in reversed(range(self.gp_vl_raw_data.count())): 
                self.gp_vl_raw_data.itemAt(i).widget().deleteLater()

            canvas, toolbar = self.graph.create_power_spectrum_plot(psd_pol0, psd_pol1, self.number_of_channels, self.f0, self.bw, 'RAW DATA')
        
            self.gp_vl_raw_data.addWidget(toolbar) 
            self.gp_vl_raw_data.addWidget(canvas)



    def display_corrected_plot(self, psd_pol0, psd_pol1, fft_corrected_pol0, fft_corrected_pol1):

        self.fft_corrected_pol0 = fft_corrected_pol0
        self.fft_corrected_pol1 = fft_corrected_pol1

        # Check if the vertical layout has already a widget (canvas and toolbar)
        # Optionally, delete the widget to free up resources
        # self.button2.deleteLater()
        if(self.gp_vl_filtered_data.count() > 0):

            for i in reversed(range(self.gp_vl_filtered_data.count())): 
                self.gp_vl_filtered_data.itemAt(i).widget().deleteLater()

            canvas, toolbar = self.graph.create_power_spectrum_plot(psd_pol0, psd_pol1, self.number_of_channels, self.f0, self.bw, 'FILTERED DATA')
        
            self.gp_vl_filtered_data.addWidget(toolbar) 
            self.gp_vl_filtered_data.addWidget(canvas)


    
    def filter_data(self):

        # convert the slider value to the 0 - 7 scale values
        threshold_pol0 = float(self.threshold_pol_left_psa_hs.value()*7/100)
        threshold_pol1 = float(self.threshold_pol_right_psa_hs.value()*7/100)

        wpoints_pol_left = int(self.wpoints_pol_left_cmb.currentText())
        wpoints_pol_right = int(self.wpoints_pol_right_cmb.currentText())


        # Create and start the worker thread, passing the parameter (skip_calibration) to it
        self.thread_psd_pol0 = DetectOutliers(self.psd_pol0, wpoints_pol_left, threshold_pol0, self.psd_pol1, wpoints_pol_right, threshold_pol1) 
        self.thread_psd_pol0.outliers_signal_psd.connect(self.get_outliers)
        # Connect the signal from the worker thread to update the start progressbar
        self.thread_psd_pol0.update_progressbar_value.connect(self.update_filter_progressbar_value)
         # Connect the signal from the worker thread to update the filter progressbar range
        self.thread_psd_pol0.update_progressbar_range.connect(self.update_filter_progressbar_range)
        # Connect the signal from the worker thread to update the buttons in the interface
        self.thread_psd_pol0.update_button_signal.connect(self.disable_buttons)
        # Connect the signal from the worker thread to update the statusbar messages
        self.thread_psd_pol0.update_statusbar_signal.connect(self.update_statusbar_message)
        # Start the thread
        self.thread_psd_pol0.start()
        # self.thread_psd_pol0.wait() # wait until the QThread is completed


    

    def get_outliers(self, outliers_pol0, outliers_pol1):

        print('Outliers_pol0:', outliers_pol0)
        self.outliers_indices_pol0 = outliers_pol0

        print('Outliers_pol1:', outliers_pol1)
        self.outliers_indices_pol1 = outliers_pol1 

        # For each outlier get a good channel of data
        channels_pol0 = self.outliers_indices_pol0.copy()
        channels_pol1 = self.outliers_indices_pol1.copy()
    
        # Create and start the worker thread, passing the parameter (skip_calibration) to it
        self.thread = CreateFilteredSpectrum(self.fft_pol0, self.fft_pol1, channels_pol0, channels_pol1, self.outliers_indices_pol0, self.outliers_indices_pol1)    
        # Connect the signal from the worker thread to update the start progressbar
        self.thread.update_progressbar_value.connect(self.update_filter_progressbar_value)
         # Connect the signal from the worker thread to update the filter progressbar range
        self.thread.update_progressbar_range.connect(self.update_filter_progressbar_range)
        # Connect the signal from the worker thread to update the buttons in the interface
        self.thread.update_button_signal.connect(self.disable_buttons)
        # Connect the signal from the worker thread to update the statusbar messages
        self.thread.update_statusbar_signal.connect(self.update_statusbar_message)
        # Connect the signal from the worker thread to get the result of the processing
        self.thread.result_signal_psd.connect(self.display_corrected_plot)
        # Start the thread
        self.thread.start()
        #self.thread.wait()



    def load_data(self, dada_file):

        print(f'Loading data for file {dada_file} just started. Please wait...')

        # Istantiate the dada file object and read the header
        dada_file_object = dada_file_manager(dada_file)
        self.dada_file_object = dada_file_object

        # Retrieve back some information from the file header
        self.f0 = float(dada_file_object.get_dict['FREQ']) # in MHz
        self.bw = float(dada_file_object.get_dict['BW']) # in MHz
        self.tsamp = float(dada_file_object.get_dict['TSAMP']) # in us
        self.tsamp = self.tsamp * 0.000001 # convert sampling time in seconds
        self.file_size = int(dada_file_object.get_dict['FILE_SIZE']) # in byte

        #self.iterations = int(self.entry_iterations_value.get())

        print('*** .dada file Header Info ***')
        print('Frequency ', self.f0)
        print('Sampling time ', self.tsamp)
        print('BandWidth ', self.bw)
        print('File Size ', self.file_size)
        print('Points per Iteration ', self.points_per_iteration)
        print('Power Spectrum channels ', self.number_of_channels)
        #print('Iterations ', self.iterations)

        # Get data from .dada file 
        dada_filename = open(dada_file,"r")
        # Skip header since already processed by the class constructor
        dada_filename.seek(4096)
        # Read the dataset
        data = np.fromfile(dada_filename, dtype='uint8', count=-1)
        # Reshape in 4 Columns [Col0:Amplitude-Pol-Left; Col1:Phase-Pol-Left; Col2: Amplitude-Pol-Right; Col3: Phase-Pol-Right]
        data_reshaped = data.reshape(int(self.file_size/4),4)
        self.data = data_reshaped
        self.max_iterations = int(len(data_reshaped)/1024)
        digit_count = len(str(abs(self.max_iterations)))
        print('*** .dada file Additional Info ***')
        print('.dada file length (rows): ', len(data_reshaped))
        print('.dada file possible iterations: ', int(len(data_reshaped)/1024)) # 1024 is the number of points per fft
        print('.dada file iterations per second: ', int(len(data_reshaped)/1024/10)) # -dada file contains 10 seconds of data

        # Set iteration label text
        self.iterations_lb.setText(f'ITERATIONS [{self.max_iterations}]')
        # Set the QlineEdit input limits
        # Set the validator to only allow digits between 0 and 9
        validator = QIntValidator(0, self.max_iterations)  # Set range 0 to max_iterations
        self.iterations_le.setValidator(validator)
        # Set maximum length of input to 5 characters
        self.iterations_le.setMaxLength(int(digit_count))
        # Enable the QlineEdit input field
        self.iterations_le.setEnabled(True)

        # Update the header
        self.filename_val.setText(str(os.path.basename(dada_file)))
        self.source_name_val.setText(str(dada_file_object.get_dict['SOURCE']))
        self.ra_val.setText(str(dada_file_object.get_dict['RA']))
        self.dec_val.setText(str(dada_file_object.get_dict['DEC']))
        self.telescope_val.setText(str(dada_file_object.get_dict['TELESCOPE']))
        self.instrument_val.setText(str(dada_file_object.get_dict['INSTRUMENT']))
        self.frequency_val.setText(str(self.f0))
        self.bandwidth_val.setText(str(self.bw))
        self.sampling_time_val.setText(str(dada_file_object.get_dict['TSAMP']))
        self.iterations_val.setText(str(self.max_iterations))
        self.interations_per_second_val.setText(str(int(self.max_iterations/10)))
        self.points_per_iteration_val.setText(str(self.points_per_iteration))
        self.spectrum_channels_val.setText(str(self.number_of_channels))

    

    def open_file_dialog(self):
        
        self.selected_data_file, _ = QFileDialog.getOpenFileName(None, "Select a .dada File","", "Dada Files (*.dada);;All Files (*)")

        if self.selected_data_file:  # Check if a file was selected
            # Check if the selected file has the .dada extension
            _, file_extension = os.path.splitext(self.selected_data_file)
            if file_extension.lower() == ".dada":
                print(f"File selected: {self.selected_data_file}")  # Print the selected file path to the console
                self.load_data(self.selected_data_file)
            else:
                self.label.setText("Selected file is not a .dada file")  # Display a message if file extension is wrong
                print("Selected file is not a .dada file")
                self.reset_buttons()
    


    def power_spectrum_builder(self):

        iterations = int(self.iterations_le.text())
        nchan = self.number_of_channels
        npoints = self.points_per_iteration
        data = self.data

        self.data_processing_start_psa_pbar.reset()
        self.data_processing_start_psa_pbar.setRange(0, iterations)

        # Create and start the worker thread, passing the parameter (skip_calibration) to it
        self.thread = CreateRawSpectrum(iterations, npoints, nchan, data)    
        # Connect the signal from the worker thread to update the start progressbar value
        self.thread.update_progressbar_value.connect(self.update_start_progressbar_value)
        # Connect the signal from the worker thread to update the start progressbar range
        self.thread.update_progressbar_range.connect(self.update_start_progressbar_range)
        # Connect the signal from the worker thread to update the buttons in the interface
        self.thread.update_button_signal.connect(self.disable_buttons)
        # Connect the signal from the worker thread to update the statusbar messages
        self.thread.update_statusbar_signal.connect(self.update_statusbar_message)
        # Connect the signal from the worker thread to get the result of the processing
        self.thread.result_signal_psd.connect(self.display_plot)
        
        # Start the thread
        self.thread.start()




    def reset_buttons(self):

        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.filter_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)


    
    def save_data(self, real_uint8_pol0, imag_uint8_pol0, real_uint8_pol1, imag_uint8_pol1):

        self.statusbar.showMessage('Saving data to disk. Please wait...')

        # Flatten and reshape into a single-column array
        real_uint8_pol0_col_reshaped = real_uint8_pol0.reshape(-1, 1) # Shape: (n, 1)
        imag_uint8_pol0_col_reshaped = imag_uint8_pol0.reshape(-1, 1) # Shape: (n, 1)
        real_uint8_pol1_col_reshaped = real_uint8_pol1.reshape(-1, 1)
        imag_uint8_pol1_col_reshaped = imag_uint8_pol1.reshape(-1, 1)

        # Stack into a 2D array with shape (3072, 2)
        merged_data = np.hstack([real_uint8_pol0_col_reshaped, imag_uint8_pol0_col_reshaped, real_uint8_pol1_col_reshaped, imag_uint8_pol1_col_reshaped])

        # Replace the corrected data in the original dataset
        self.data[0:len(merged_data)] = merged_data[:]

        data_length = len(self.data)

        # _p stands for "processed"
        file_name_p = self.destination_path + '/' + self.dada_file_object.get_fbasename[0] + "_p" + self.dada_file_object.get_fextention

        print('Writing header to file ' + file_name_p + ' ...')
        self.dada_file_object.write_DADA_header(file_name_p, self.dada_file_object.get_dict)
        print('Writing data to file ' + file_name_p + ' ...')
        self.dada_file_object.write_DADA_data(file_name_p, self.data, data_length)
        #dada_obj.write_DADA_data(file_name_p, original_dada_data, data_length)

        self.statusbar.showMessage('Saving data completed.')
        

       
    def save_file_dialog(self):
    
    # Open the folder selection dialog
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "Select a Folder",
            "",  # You can set a starting directory here
            QFileDialog.ShowDirsOnly
        )
            
        # Print the selected folder path
        if folder_path:
            
            print("Selected folder:", folder_path)
            self.destination_path = folder_path

            # Create and start the worker thread, passing the parameter (skip_calibration) to it
            self.thread_time_domain_corrected = CreateFilteredTimeDomainSignal(self.fft_corrected_pol0, self.fft_corrected_pol1, 
                self.outliers_indices_pol0, self.outliers_indices_pol1)
            self.thread_time_domain_corrected.update_statusbar_signal.connect(self.update_statusbar_message)
            self.thread_time_domain_corrected.result_signal_filtered_time_domain.connect(self.save_data)
            # Start the thread
            self.thread_time_domain_corrected.start()
            #self.thread_time_domain_corrected.wait() # wait until the QThread is completed
        
        else:
            
            print("No folder selected.")
            self.statusbar.showMessage('No folder selected. Data can not be saved.')



    def slider_changed_value_pol_left(self):

        value = round(self.MAX_SLIDER_VAL/100*self.threshold_pol_left_psa_hs.value(),1)
        self.threshold_pol_left_lb.setText(f'THRESHOLD [{value}]')

    
    
    def slider_changed_value_pol_right(self):

        value = round(self.MAX_SLIDER_VAL/100*self.threshold_pol_right_psa_hs.value(),1)
        self.threshold_pol_right_lb.setText(f'THRESHOLD [{value}]')


    
    def update_filter_progressbar_range(self, min_value: int, max_value: int):

        # setting min_value = 0 and max_value = 0, makes the bar show a busy indicator
        self.data_processing_filter_psa_pbar.setRange(min_value, max_value)  



    
    def update_filter_progressbar_value(self, value: int):

        self.data_processing_filter_psa_pbar.setValue(value)
    
    

    def update_start_progressbar_range(self, min_value: int, max_value: int):

        # setting min_value = 0 and max_value = 0, makes the bar show a busy indicator
        self.data_processing_start_psa_pbar.setRange(min_value, max_value)  


    
    def update_start_progressbar_value(self, value: int):

        self.data_processing_start_psa_pbar.setValue(value)
   


    def update_statusbar_message(self, value: str):

        self.statusbar.showMessage(value)

       
    
        
    def reset_data(self):

        # Remove plots from the graph areas
        if(self.gp_vl_raw_data.count() > 0):

            for i in reversed(range(self.gp_vl_raw_data.count())): 
                self.gp_vl_raw_data.itemAt(i).widget().deleteLater()

        if(self.gp_vl_filtered_data.count() > 0):

            for i in reversed(range(self.gp_vl_filtered_data.count())): 
                self.gp_vl_filtered_data.itemAt(i).widget().deleteLater()

        # Add an empty plot in the plot area
        # self.graph = GraphCreator()
        canvas_raw_data = self.graph.create_MPL_empty_plot('RAW DATA')
        canvas_filtered_data = self.graph.create_MPL_empty_plot('FILTERED DATA')
        self.gp_vl_raw_data.addWidget(canvas_raw_data)
        self.gp_vl_filtered_data.addWidget(canvas_filtered_data)

        # Update the header
        self.filename_val.setText('<Unknown>')
        self.source_name_val.setText('<Unknown>')
        self.ra_val.setText('<Unknown>')
        self.dec_val.setText('<Unknown>')
        self.telescope_val.setText('<Unknown>')
        self.instrument_val.setText('<Unknown>')
        self.frequency_val.setText('<Unknown>')
        self.bandwidth_val.setText('<Unknown>')
        self.sampling_time_val.setText('<Unknown>')
        self.iterations_val.setText('<Unknown>')
        self.interations_per_second_val.setText('<Unknown>')
        self.points_per_iteration_val.setText('<Unknown>')
        self.spectrum_channels_val.setText('<Unknown>')

        # Reset data variables
        self.data = None # variable containing the whole dataset reshaped (n raws x 4 columns) which is loaded with the button 'Load'
        self.dada_file_object = None
        
        self.psd_pol0 = None # the raw power spectrum density for pol0
        self.psd_pol1 = None # the raw power spectrum density for pol1
        self.fft_pol0 = None # the raw fft coefficients for pol0
        self.fft_pol1 = None # the raw fft coefficients for pol1
        self.outliers_indices_pol0 = None
        self.outliers_indices_pol1 = None

        self.fft_corrected_pol0 = None
        self.fft_corrected_pol1 = None

        self.destination_path = ""

        # Reset button status
        self.reset_buttons()
        
        # Reset filter widgets
        self.data_processing_start_psa_pbar.setValue(0)
        self.data_processing_filter_psa_pbar.setValue(0)

        # Set visible item of the combobox dropdown list 
        self.wpoints_pol_left_cmb.setCurrentIndex(4)
        self.wpoints_pol_right_cmb.setCurrentIndex(4)
       
        # Set threshold values
        self.threshold_pol_left_psa_hs.setValue(int(round(3*100/self.MAX_SLIDER_VAL))) # 7 corresponds to 100
        self.threshold_pol_right_psa_hs.setValue(int(round(3*100/self.MAX_SLIDER_VAL))) # 7 corresponds to 100

        # Set Iteration label and textfield
        self.iterations_lb.setText('ITERATIONS [0]')
        self.iterations_le.setEnabled(False)
        self.iterations_le.setText('')

        self.statusbar.showMessage('')
       


    
           