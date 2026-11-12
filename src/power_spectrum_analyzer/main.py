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

from power_spectrum_analyzer.processing.power_spectrum_analyzer import PowerSpectrumAnalyzer


class MainUI(QMainWindow):

    def __init__(self):
        # Create the parser
        parser = argparse.ArgumentParser(description="Run the script with optional debug mode")
        # Add the -d / --debug option
        parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode")
        # Parse command line arguments
        args = parser.parse_args()

        if args.debug:
            print("DEBUG MODE: ON")
            self.debug_on = True
        else:
            print("DEBUG MODE: OFF")
            self.debug_on = False

        # Get the directory where the main script is located
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        print("Script is located at:", self.script_directory)

        # Read the first valid data directory from config.ini
        default_data_dir = self.get_data_directory(self.script_directory, "config.ini")

        super(MainUI, self).__init__()

        # Constructing the absolute path dynamically to the .ui file located within the 'gui' folder
        ui_path = os.path.join(self.script_directory, "gui", "power_spectrum_analyzer_gui.ui")
        uic.loadUi(ui_path, self)

        # Set the fixed size of the window (width, height)
        self.setFixedSize(1500, 960)

        # Pass the .ui widgets and default_data_dir to PowerSpectrumAnalyzer for initialization
        self.power_spectrum_analyzer = PowerSpectrumAnalyzer(
            self.load_btn, self.save_btn, self.start_btn,
            self.filter_btn, self.reset_btn, self.wpoints_pol_left_cmb, self.wpoints_pol_right_cmb,
            self.threshold_pol_left_psa_hs, self.threshold_pol_right_psa_hs, self.iterations_le, self.data_processing_method_psa_cb,
            self.data_processing_start_psa_pbar, self.data_processing_filter_psa_pbar, self.threshold_pol_left_lb, self.threshold_pol_right_lb,
            self.iterations_lb, self.gp_vl_raw_data, self.gp_vl_filtered_data, self.source_name_val, self.ra_val, self.dec_val,
            self.telescope_val, self.instrument_val, self.frequency_val, self.bandwidth_val, self.sampling_time_val, self.iterations_val,
            self.iterations_per_second_val, self.points_per_iteration_val, self.spectrum_channels_val, self.filename_val, self.statusbar,
            self.advanced_btn, default_data_dir=default_data_dir
        )

    def get_data_directory(self, app_path, config_filename="config.ini"):
        config_path = os.path.join(app_path, config_filename)
        fallback_dir = os.path.expanduser("~")

        if not os.path.exists(config_path):
            print(f"Warning: '{config_path}' not found. Using home directory: {fallback_dir}")
            return fallback_dir

        config = configparser.ConfigParser()
        config.read(config_path)

        if config.has_option('Paths', 'data'):
            raw_paths = config.get('Paths', 'data').strip()
            if raw_paths:
                # Extract only the first directory if multiple paths are provided
                first_dir = raw_paths.replace('\n', ',').split(',')[0].strip()
                if os.path.isdir(first_dir):
                    print(f"Loaded default data directory from config.ini: {first_dir}")
                    return first_dir
                else:
                    print(f"Warning: Directory '{first_dir}' in config.ini does not exist. Falling back to {fallback_dir}")

        return fallback_dir


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setStyleSheet('QComboBox {combobox-popup: 0}')

    window = MainUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()




    
