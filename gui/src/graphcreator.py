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
#from PyQt5 import QtCore

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

    def retrieve_x_axis(self, graph_type, header_info):

        # The rest frequency is used instead of the central frequency (as CLASS does too)

        central_frequency = header_info["central_frequency"]
        channels = header_info["channels"]
        bandwidth = header_info["bandwidth"]
        doppler_speed = header_info["dopler_speed"] 
        frequency_resolution =  header_info["frequency_resolution"]
        observer_velocity = header_info["observer_velocity"][0]
        radial_velocity = header_info["radial_velocity"]
        rest_frequency = header_info["rest_frequency"][0]
        speed_resolution = header_info["speed_resolution"] # the channel resolution in km/s calculated using the formula -> delta_v = (bw/cf)*c with c = 300000 km/s

        x_axis = [] # the x axis described by values and unit

        x = None # the x axis values
        u = None # the unit relative to the x axis

        # compute the x axis according to selected the graph type 

        if graph_type == 'Channel':
            
            x = np.arange(0, channels, 1) # min, max, step
            u = 'Channel [#]'
        
        if graph_type == 'Rest Frequency':

            print('Central rest frequency [MHz]', rest_frequency)
            print('Bandwidth [MHz]', bandwidth)
            print('Frequency resolution [GHz]', frequency_resolution)
            # Better of using linespace in case the binning is a non integer (such as the channel resolution)
            # It was used to consider the local oscillator frequency value but according to CLASS, the rest frequency it is used
            # The observation frequency (whatever it is i.e. local oscillator or rest frequency) is considered in the middle of the bandwidth
            x = np.linspace(rest_frequency-bandwidth/2, rest_frequency+bandwidth/2, num=channels) # min, max, n. channels. bw_ghz/2 represtents the half bandwidth
            # If the observation frequency is at the beginning of the bandwidth then
            # x = np.linspace(rest_frequency, rest_frequency + bandwidth, channels) # linespace uses the total number of bins
 
            u = 'Frequency [MHz]'

        if graph_type == 'Sky Frequency':

            """
            Calculate the observed frequency based on the rest frequency, source speed, and speed of light.
    
            Parameters:
            - f_rest: Rest frequency of the source (Hz)
            - v: Speed of the source relative to the observer (m/s)
            - c: Speed of light in vacuum (m/s), default is 3e8 m/s
    
            Returns:
            - f_observed: Observed frequency (Hz)
            """
            
            # Doppler effect formula for frequency shift
            f_observed = rest_frequency * math.sqrt((1 - observer_velocity / CLIGHT) / (1 + observer_velocity / CLIGHT))
            

            x = np.linspace( f_observed-bandwidth/2,  f_observed+bandwidth/2, num=channels)
            u = 'Frequency [MHz]'

        if graph_type == 'Velocity':

            # the whole bandwidth in km/s is equal to channels*speed_resolution
            # we set the zero km/s at the middle of the bandwidth
            # we also need to take into account the radial velocity and sum it algebrically sum to the limits 
            # It is a composition of velocities i.e. Galileian sum of velocities

            x = np.linspace(-(speed_resolution*channels/2) + radial_velocity, +(speed_resolution*channels/2) + radial_velocity, num=channels) # min, max, steps
            u = 'Velocity [km/s]'
            
        x_axis.append(x)
        x_axis.append(u)

        return x_axis

    def createMPLPlot(self, mode, feeds, axis, chmin, chmax, skip_calibration, data, header_info):

        # According to the graph mode, create the plot
        if(mode == 'Position Switching'):

            fig, ax = plt.subplots(figsize=(10, 8), facecolor=(241/255, 248/255, 255/255))

            # Add a grid to the plot
            ax.grid(True)
            # Set the title for the x-axis
            #ax.set_title('Click on legend line to toggle line on/off')
            ax.set_xlabel(axis[1])
            # Set the title for the y-axis
            ax.set_ylabel(self.get_y_axis_label(skip_calibration))
            # Set the background color of the plot area (axes) using RGB values
            ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)

            (line1, ) = ax.step(axis[0][chmin:chmax], data[0][chmin:chmax], where='mid', label='[FEED '+feeds[0]+'] LL', linewidth=1)
            (line2, ) = ax.step(axis[0][chmin:chmax], data[1][chmin:chmax], where='mid', label='[FEED '+feeds[0]+'] RR', linewidth=1)

            # In case we have an inverted x-axis then MPL will invert it automatically to have negative numbers before
            # In this case, if we want at first the positive numbers, we need to invert them manually

            if(axis[0][0] > axis[0][chmax-1]):
                # Invert the x-axis to have positive values on the left
                plt.gca().invert_xaxis()

            leg = ax.legend(fancybox=True, shadow=True)
            
            lines = [line1, line2]
            map_legend_to_ax = {}  # Will map legend lines to original lines.

            pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

            for legend_line, ax_line in zip(leg.get_lines(), lines):
                legend_line.set_picker(pickradius)  # Enable picking on the legend line.
                map_legend_to_ax[legend_line] = ax_line


            def on_pick(event):
                # On the pick event, find the original line corresponding to the legend
                # proxy line, and toggle its visibility.
                legend_line = event.artist

                # Do nothing if the source of the event is not a legend line.
                if legend_line not in map_legend_to_ax:
                    return

                ax_line = map_legend_to_ax[legend_line]
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled.
                legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
            

            fig.canvas.mpl_connect('pick_event', on_pick)

            # Works even if the legend is draggable. This is independent from picking legend lines.
            leg.set_draggable(True)
            # Create a canvas widget for the plot
            canvas = FigureCanvas(fig)
            # Create a toolbar for the plot
            toolbar = NavigationToolbar2QT(canvas)

            # to delete an unwanted toolbar button get at first the index
            # print(toolbar.toolitems)
            # next -> delete for instance, the button at index 3 (del self.toolitems[3])
            
            return canvas, toolbar

        else:

            #x_val = np.asarray(axis[0][chmin:chmax], dtype="object")
            #y_val = np.asarray(data[0][chmin:chmax], dtype="object")

            fig, ax = plt.subplots(figsize=(10, 8), facecolor=(241/255, 248/255, 255/255))

            # Add a grid to the plot
            ax.grid(True)
            # Set the title for the x-axis
            #ax.set_title('Click on legend line to toggle line on/off')
            ax.set_xlabel(axis[1])
            # Set the title for the y-axis
            ax.set_ylabel(self.get_y_axis_label(skip_calibration))
            # Set the background color of the plot area (axes) using RGB values
            ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)

            # Get the counts and edges for the histogram using numpy's histogram function
            # counts, edges = np.histogram(data[0][chmin:chmax], bins=axis[0][chmin:chmax])
            
            # (line1, ) = ax.plot(axis[0][chmin:chmax], data[0][chmin:chmax], marker='_', markersize=10, alpha=0.7, linewidth=1, label='['+feeds[0]+'] LL')
            # Use step plot to simulate the histogram

            (line1, ) = ax.step(axis[0][chmin:chmax], data[0][chmin:chmax], where='mid', label='[FEED '+feeds[0]+'] LL', linewidth=1)
            (line2, ) = ax.step(axis[0][chmin:chmax], data[1][chmin:chmax], where='mid', label='[FEED '+feeds[0]+'] RR', linewidth=1)
            (line3, ) = ax.step(axis[0][chmin:chmax], data[2][chmin:chmax], where='mid', label='[FEED '+feeds[1]+'] LL', linewidth=1)
            (line4, ) = ax.step(axis[0][chmin:chmax], data[3][chmin:chmax], where='mid', label='[FEED '+feeds[1]+'] RR', linewidth=1)

            # In case we have an inverted x-axis then MPL will invert it automatically to have negative numbers before
            # In this case, if we want at first the positive numbers, we need to invert them manually

            if(axis[0][0] > axis[0][chmax-1]):
                # Invert the x-axis to have positive values on the left
                plt.gca().invert_xaxis()

            leg = ax.legend(fancybox=True, shadow=True)
            
            lines = [line1, line2, line3, line4]
            map_legend_to_ax = {}  # Will map legend lines to original lines.

            pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

            for legend_line, ax_line in zip(leg.get_lines(), lines):
                legend_line.set_picker(pickradius)  # Enable picking on the legend line.
                map_legend_to_ax[legend_line] = ax_line


            def on_pick(event):
                # On the pick event, find the original line corresponding to the legend
                # proxy line, and toggle its visibility.
                legend_line = event.artist

                # Do nothing if the source of the event is not a legend line.
                if legend_line not in map_legend_to_ax:
                    return

                ax_line = map_legend_to_ax[legend_line]
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled.
                legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
            

            fig.canvas.mpl_connect('pick_event', on_pick)

            # Works even if the legend is draggable. This is independent from picking legend lines.
            leg.set_draggable(True)
            # Create a canvas widget for the plot
            canvas = FigureCanvas(fig)
            # Create a toolbar for the plot
            toolbar = NavigationToolbar2QT(canvas)

            # to delete an unwanted toolbar button get at first the index
            # print(toolbar.toolitems)
            # next -> delete for instance, the button at index 3 (del self.toolitems[3])
            
            return canvas, toolbar

            # plt.draw() to update the plot without regenerating
            # plt.show()


    def create_plot(self, mode, feeds, axis, chmin, chmax, skip_calibration, data, header_info):

        # Generic function to work create plots

        line = [None, None, None, None]
        labels = []
        lines = []

        print(len(data))
        print(type(data))
      
        if(len(data) > 1): # case of averages for each feed and polarization
      
            print("The array contains more than one list (2D array).")

            # Construct the plot labels according to the number of feeds
            for i in range(len(feeds)):

                labels.append('[FEED ' + feeds[i] + '] LL')
                labels.append('[FEED ' + feeds[i] + '] RR')

        else:
            
            print("The array does not contain more than one list.")

            # Construct a single plot label (case of total average for all feeds and polarizations)
            labels.append('[TOTAL AVG]')

            array_dim = 1

        fig, ax = plt.subplots(figsize=(10, 8), facecolor=(241/255, 248/255, 255/255))

        # Add a grid to the plot
        ax.grid(True)
        # Set the title for the x-axis
        #ax.set_title('Click on legend line to toggle line on/off')
        ax.set_xlabel(axis[1])
        # Set the title for the y-axis
        ax.set_ylabel(self.get_y_axis_label(skip_calibration))
        # Set the background color of the plot area (axes) using RGB values
        ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)

        for i in range(len(data)):

            (line[i], ) = ax.step(axis[0][chmin:chmax], data[i][chmin:chmax], where='mid', label=labels[i], linewidth=1)

            lines.append(line[i]) 

        # In case we have an inverted x-axis then MPL will invert it automatically to have negative numbers before
        # In this case, if we want at first the positive numbers, we need to invert them manually

        if(axis[0][0] > axis[0][chmax-1]):
            # Invert the x-axis to have positive values on the left
            plt.gca().invert_xaxis()

        leg = ax.legend(fancybox=True, shadow=True)

        map_legend_to_ax = {}  # Will map legend lines to original lines.

        pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

        for legend_line, ax_line in zip(leg.get_lines(), lines):
            legend_line.set_picker(pickradius)  # Enable picking on the legend line.
            map_legend_to_ax[legend_line] = ax_line


        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist

            # Do nothing if the source of the event is not a legend line.
            if legend_line not in map_legend_to_ax:
                return

            ax_line = map_legend_to_ax[legend_line]
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()
            
        fig.canvas.mpl_connect('pick_event', on_pick)

        # Works even if the legend is draggable. This is independent from picking legend lines.
        leg.set_draggable(True)
        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)
        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)

        # to delete an unwanted toolbar button get at first the index
        # print(toolbar.toolitems)
        # next -> delete for instance, the button at index 3 (del self.toolitems[3])
        
        return canvas, toolbar

    
    
    def create_avg_raw_data_plot(self, data, pol_type, feeds):

        # reset all plots previously created, if any
        plt.close('all')
        
        if(pol_type == 'LCP-RCP'):
        
            line = [None] * (len(feeds)*2)

        else:

            line = [None] * len(feeds)
        
        labels = []
        lines = []

       
        # Create the plot labels according to the number of selected feeds
        for i in range(len(feeds)):

            if(pol_type == 'LCP-RCP'):

                labels.append(f'FEED [{feeds[i]}-LCP]')
                labels.append(f'FEED [{feeds[i]}-RCP]')

            else:
                
                labels.append(f'FEED [{feeds[i]}]')


        fig, ax = plt.subplots(figsize=(10, 8), facecolor=(241/255, 248/255, 255/255))

        # Add a grid to the plot
        ax.grid(True)
        # Set the title for the x-axis
        ax.set_title(f'AVG RAW DATA [{pol_type}]')
        ax.set_xlabel("Channel [#]")
        # Set the title for the y-axis
        ax.set_ylabel("Amplitude [AU]")
        # Set the background color of the plot area (axes) using RGB values
        ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)

        
        print(len(labels))
        print(len(data))
        print(len(data[0]))
       

        # Create x-axis values ranging from 0 to 1023
        x = np.arange(0, len(data[0]))  # Creates an array from 0 to 1023


        for i in range(len(labels)):

            if(len(data[0]) == 1): # The data array contains a single number

                # Plotting the single number
                (line[i], ) = ax.plot(data[i], 'o', label=labels[i], linewidth=1)  # 'o' indicates a marker for the point
            
            else:   
                
                (line[i], ) = ax.step(x, data[i], where='mid', label=labels[i], linewidth=1)

            lines.append(line[i]) 
        
        leg = ax.legend(fancybox=True, shadow=True)

        map_legend_to_ax = {}  # Will map legend lines to original lines.

        pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

        for legend_line, ax_line in zip(leg.get_lines(), lines):
            legend_line.set_picker(pickradius)  # Enable picking on the legend line.
            map_legend_to_ax[legend_line] = ax_line
        
        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist

            # Do nothing if the source of the event is not a legend line.
            if legend_line not in map_legend_to_ax:
                return

            ax_line = map_legend_to_ax[legend_line]
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

        # Works even if the legend is draggable. This is independent from picking legend lines.
        leg.set_draggable(True)
        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)
        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)

        # to delete an unwanted toolbar button get at first the index
        # print(toolbar.toolitems)
        # next -> delete for instance, the button at index 3 (del self.toolitems[3])

        return canvas, toolbar




    
    def create_total_power_cross_scan_plot(self, time, data, pol_type, feeds):

        # reset all plots previously created, if any
        plt.close('all')
        
        if(pol_type == 'LCP-RCP'):
        
            line = [None] * (len(feeds)*2)

        else:

            line = [None] * len(feeds)
        
        labels = []
        lines = []

        #print(feeds)
        #print(time)
        #print(data)

        # Create the plot labels according to the number of selected feeds
        for i in range(len(feeds)):

            if(pol_type == 'LCP-RCP'):

                labels.append(f'FEED [{feeds[i]}-LCP]')
                labels.append(f'FEED [{feeds[i]}-RCP]')

            else:
                
                labels.append(f'FEED [{feeds[i]}]')
    
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=(241/255, 248/255, 255/255))

        # Add a grid to the plot
        ax.grid(True)
        # Set the title for the x-axis
        ax.set_title(f'Cross-Scan [{pol_type}]')
        ax.set_xlabel("Time [s]")
        # Set the title for the y-axis
        ax.set_ylabel("Amplitude [AU]")
        # Set the background color of the plot area (axes) using RGB values
        ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)



        for i in range(len(data)):

            (line[i], ) = ax.step(time, data[i], where='mid', label=labels[i], linewidth=1)

            lines.append(line[i]) 
        
        leg = ax.legend(fancybox=True, shadow=True)

        map_legend_to_ax = {}  # Will map legend lines to original lines.

        pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

        for legend_line, ax_line in zip(leg.get_lines(), lines):
            legend_line.set_picker(pickradius)  # Enable picking on the legend line.
            map_legend_to_ax[legend_line] = ax_line
        
        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist

            # Do nothing if the source of the event is not a legend line.
            if legend_line not in map_legend_to_ax:
                return

            ax_line = map_legend_to_ax[legend_line]
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

             
        fig.canvas.mpl_connect('pick_event', on_pick)

        # Works even if the legend is draggable. This is independent from picking legend lines.
        leg.set_draggable(True)
        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)
        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)

        # to delete an unwanted toolbar button get at first the index
        # print(toolbar.toolitems)
        # next -> delete for instance, the button at index 3 (del self.toolitems[3])

        return canvas, toolbar

   

    def create_total_power_map_plot(self, x_min_val, x_max_val, x_bins, y_min_val, y_max_val, y_bins, data_matrix, graph_type):

        print('Map parameter:', x_min_val, x_max_val, x_bins, y_min_val, y_max_val, y_bins)

        # Define the grid with custom limits: x and y from -3.5 to 10.5
        x = np.linspace(y_min_val, y_max_val, y_bins)
        y = np.linspace(x_min_val, x_max_val, x_bins)

        # Alternatively
        # x = np.arange(el_min_val, el_max_val, el_bins)  # Custom x axis from 
        # y = np.arange(az_min_val, az_max_val, az_bins)  # Custom y axis from 

        # Create a meshgrid from the x and y ranges
        #x_, y_ = np.meshgrid(x, y)
      
        # Create the figure holding the 2D plot
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=(241/255, 248/255, 255/255))
        ax.set_facecolor((241/255, 248/255, 255/255))



        # Remove the background for better plotting
        # Exclude zeros and find the minimum value from the grid of data
        #print('Matrix subtracted max value: ', data_matrix[data_matrix != 0].max())
        #min_value = data_matrix[data_matrix != 0].min()
        #print('Matrix subtracted min value: ', min_value)

        # Find the maximum value in the grid
        #max_value = np.max(data_matrix)
        # Shift the background to zero
        #data_matrix[data_matrix != 0] -= min_value


        cmap = plt.cm.plasma  # You can use any continuous colormap like 'plasma', 'inferno', etc.
        #norm = colors.Normalize(vmin=data_matrix.min(), vmax=data_matrix.max())  # Normalize to the min/max of your data

        #cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

        # Create the 2D plot
        # For a contour map use:
        # cp = plt.contourf(x, y, data_matrix, levels=100, cmap=cmap, vmin=0, vmax=max_value-min_value, alpha = 1)
     
        # For a pixelated map use: 
        cp = ax.imshow(data_matrix, cmap=cmap, interpolation='none', extent=[y_min_val, y_max_val, x_min_val, x_max_val], alpha = 1)
        
        # Create a colorbar with 100 levels
        #cbar = fig.colorbar(cp, ax=ax, ticks=np.linspace(np.min(data_matrix), np.max(data_matrix), 10), orientation='vertical')

        '''
        # Set tick labels to grad (degrees -> grades -> DMS)
        x_ticks = ax.get_xticks()  # Get current ticks
        new_x_ticks = [self.rad_to_grads_minutes_seconds(tick) for tick in x_ticks]  # Convert radians to grads
        #new_ticks_in_grads = [self.degrees_to_grades(tick) for tick in new_ticks]  # Convert degrees to grads

        # Format ticks as grad, minutes, and seconds
        #formatted_ticks = [self.grades_to_dms(tick) for tick in new_ticks_in_grads]

        # Apply the formatted tick labels to the plot
        ax.set_xticklabels(new_x_ticks)

        y_ticks = ax.get_yticks()
        new_y_ticks = [self.rad_to_grads_minutes_seconds(tick) for tick in y_ticks]  # Convert radians to grads
        # Apply the formatted tick labels to the plot
        ax.set_yticklabels(new_y_ticks)'''


        cbar = fig.colorbar(cp,orientation="vertical")

        # 5) Add labels and title
        if(graph_type == 'RA-DEC'):

            plt.ylabel('DEC [rad]')
            plt.xlabel('RA [rad]')
        
        else:

            plt.ylabel('EL [rad]')
            plt.xlabel('AZ [rad]')

        plt.title('Cross-Scan MAP')

        #ax.set_aspect(0.3)
        canvas = FigureCanvas(fig)
        ax.set_aspect('auto') # this is important to keep a full size figure inside the canvas

        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)
        #toolbar = None

        return canvas, toolbar




    def get_y_axis_label(self, skip_calibration):

        y_axis_label = ""

        if(skip_calibration):

            y_axis_label = "Counts"

        else:

            y_axis_label = "Antenna Temperature [K]"

        return y_axis_label 

    def get_plot(self):

        # Create a canvas widget for the plot
        canvas = FigureCanvas(self.fig)
        canvas.draw()  # Important: triggers layou

        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)
            
        return canvas, toolbar

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
      
        '''
        fig = plt.figure(figsize=(2, 2), facecolor='lightskyblue',
                 layout='constrained')
        fig.suptitle('Figure')
        ax = fig.add_subplot()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Distance [km]')
        ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')'''

        '''
        fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
        np.random.seed(19680801)
        t = np.arange(200)
        x = np.cumsum(np.random.randn(200))
        y = np.cumsum(np.random.randn(200))
        linesx = ax.plot(t, x, label='Random walk x')
        linesy = ax.plot(t, y, label='Random walk y')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Distance [km]')
        ax.set_title('Random walk example')
        ax.legend()'''

        


        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)

        return canvas

        # Show the plot
        #plt.show()


    
    def rad_to_grads_minutes_seconds(self, radians):
        
        # Convert radians to degrees
        degrees = radians * 180 / math.pi
        
        # Convert degrees to grads
        grads = degrees * 1.1111  # 1 degree = 1.1111 grads
        
        # Get the integer part and fractional part for grads
        grad_deg = int(grads)
        grad_min = (grads - grad_deg) * 60
        grad_min_int = int(grad_min)
        grad_sec = (grad_min - grad_min_int) * 60
        
        # Format the result
        result = f"{grad_deg}° {grad_min_int}' {grad_sec:.2f}\""
        return result

    
    
    def create_power_spectrum_plot(self, psd_pol0, psd_pol1, nchan, f0, bw, title):

        samplingFrequency = nchan
        samplingInterval = 1 / nchan
        freq = np.fft.fftfreq(samplingFrequency, samplingInterval)
        freq_abs_val = np.fft.fftshift((freq/1024*bw)+f0)

        # reset all plots previously created, if any
        plt.close('all')

        line = [None] * 2 # Store the datasets in the array
        lines = []

        #fig = plt.figure(figsize=(5, 3), facecolor=(241/255, 248/255, 255/255))


        fig = plt.figure(figsize=(5, 3), dpi=100, facecolor=(241/255, 248/255, 255/255), layout='constrained')
        fig.suptitle('')
        ax = fig.add_subplot()
      

        # Add a grid to the plot
        ax.grid(True)
        # Set the title for the x-axis
        ax.set_title(title)
        ax.set_xlabel("Frequency [MHz]")
        # Set the title for the y-axis
        ax.set_ylabel("Amplitude [AU]")
        # Set the background color of the plot area (axes) using RGB values
        ax.set_facecolor((241/255, 248/255, 255/255))  # Light blue background for the plot area (RGB: 0.9, 0.9, 1)

        # Create x-axis values ranging from 0 to 1023. 1024 channels in total
        x = np.arange(0, 1024) 

        for i in range(1):
             
            (line[i], ) = ax.step(freq_abs_val, np.fft.fftshift(np.array(psd_pol0).real), where='mid', label='POL [LEFT]', linewidth=1)
            lines.append(line[i]) 
            (line[i+1], ) = ax.step(freq_abs_val, np.fft.fftshift(np.array(psd_pol1).real), where='mid', label='POL [RIGHT]', linewidth=1)
            lines.append(line[i+1]) 

        
        leg = ax.legend(fancybox=True, shadow=True)

        map_legend_to_ax = {}  # Will map legend lines to original lines.

        pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

        for legend_line, ax_line in zip(leg.get_lines(), lines):
            legend_line.set_picker(pickradius)  # Enable picking on the legend line.
            map_legend_to_ax[legend_line] = ax_line
        
        def on_pick(event):
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist

            # Do nothing if the source of the event is not a legend line.
            if legend_line not in map_legend_to_ax:
                return

            ax_line = map_legend_to_ax[legend_line]
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            legend_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

        # Works even if the legend is draggable. This is independent from picking legend lines.
        leg.set_draggable(True)
        # Create a canvas widget for the plot
        canvas = FigureCanvas(fig)
        # Create a toolbar for the plot
        toolbar = NavigationToolbar2QT(canvas)

        # to delete an unwanted toolbar button get at first the index
        # print(toolbar.toolitems)
        # next -> delete for instance, the button at index 3 (del self.toolitems[3])

        return canvas, toolbar
