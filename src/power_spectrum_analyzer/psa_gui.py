import configparser
import argparse
import os
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# Add the subfolder to sys.path
# sys.path.append(os.path.join(SCRIPT_DIR, 'src')) # It allows to call modules without specifying the subfolders where they are located

from power_spectrum_analyzer.processing.power_spectrum_analyzer import PowerSpectrumAnalyzer

class MainUI(QMainWindow):

    def __init__(self):

        # Create the parser
        parser = argparse.ArgumentParser(description="Run the script with optional debug mode")
        # Add the -d / --debug option
        parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode") # if '-d' then store 'True' value in args
        # Parse command line arguments
        args = parser.parse_args()
        # Check if the debug flag is set and print accordingly. If so all projects id are displayed in the combo box otherwise only that referred to the username is loaded
        self.debug_on = None

        if args.debug:
            print("DEBUG MODE: ON")
            self.debug_on = True
        else:
            print("DEBUG MODE: OFF")
            self.debug_on = False

        # Backends and relative IP addresses of the servers used to store the raw data   
        # SERVER_BACKEND = ['SARDARA_BKD', 'SKARAB_BKD']
        # SERVER_PING = ['192.168.200.216', '192.168.203.36']

        # Get the directory where the main script is located
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        print("Script is located at:", self.script_directory)

        super(MainUI, self).__init__()

        # Constructing the absolute path dinamically to the file .ui located within the folder 'gui'
        ui_path = os.path.join(self.script_directory, "gui", "power_spectrum_analyzer_gui.ui")
        uic.loadUi(ui_path, self)
        

        # Set the fixed size of the window (width, height)
        self.setFixedSize(1500, 960)  # Set the size to 800x600 pixels


        # Pass the .ui widgets to the PowerSpectrumAnalyzer for initialization
        self.power_spectrum_analyzer = PowerSpectrumAnalyzer(self.load_btn, self.save_btn, self.start_btn,
            self.filter_btn, self.reset_btn, self.wpoints_pol_left_cmb, self.wpoints_pol_right_cmb,
            self.threshold_pol_left_psa_hs, self.threshold_pol_right_psa_hs, self.iterations_le, self.data_processing_method_psa_cb,
            self.data_processing_start_psa_pbar, self.data_processing_filter_psa_pbar, self.threshold_pol_left_lb, self.threshold_pol_right_lb,
            self.iterations_lb, self.gp_vl_raw_data, self.gp_vl_filtered_data, self.source_name_val, self.ra_val, self.dec_val,
            self.telescope_val, self.instrument_val, self.frequency_val, self.bandwidth_val, self.sampling_time_val, self.iterations_val,
            self.iterations_per_second_val, self.points_per_iteration_val, self.spectrum_channels_val, self.filename_val, self.statusbar)
        
        
       
    
    '''
    def check_config_exists(self, app_path, config_ini_filename):

        paths = None
        
        full_path_config_ini = app_path + '/' + config_ini_filename
        # if config.ini exists then load the mounted drives, otherwise create the config.ini with the list of mounted drives [it should be one drive per backend]
        
        if os.path.exists(full_path_config_ini):
            
            print(f"The file '{full_path_config_ini}' exists. Retrieving mounted drives...")
            # Read out the mounted drives from the config.ini file
            # Create a ConfigParser object
            config = configparser.ConfigParser()
            # Read the configuration file
            config.read(full_path_config_ini)
            # Get the comma-separated string of paths under the 'Paths' section
            paths_str = config.get('Paths', 'PATH')
            # Split the string into a list of paths
            paths = paths_str.split(',')
        
        else:
            
            paths = ["/roach2_nuraghe/data/", "/discos-archive/data/"] 

            print(f"The file '{full_path_config_ini}' does not exist. Initialization started...")
            # config.ini initialization
            # Create a ConfigParser object
            config = configparser.ConfigParser()
            # Add a section for PATH
            config.add_section('Paths')
            # Write the list of paths under the 'Paths' section, as a comma-separated string
            config.set('Paths', 'PATH', ','.join(paths))
            # Write the configuration to a file
            with open(full_path_config_ini, 'w') as configfile:
                config.write(configfile)

        return paths


    def switch_panel(self, index):
         
        self.stackedWidget_main.setCurrentIndex(index)
'''



def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # Soluzione al mismatch tra designer sizes e monitor sizes
    app = QApplication(sys.argv)
    app.setStyleSheet('QComboBox {combobox-popup: 0}')

    window = MainUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    

