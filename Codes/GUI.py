from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QCheckBox, QPushButton, QDialog, QDialogButtonBox, QFormLayout, QFileDialog, 
    QMessageBox, QAction, QSlider, QLabel, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sys
import numpy as np
from scipy import io
from scipy import interpolate
import os
from lookup import lookup
from graph import plot_array
from matplotlib.axis import Axis 
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gm/ID Plotting tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data
        self.nch_data = None
        self.pch_data = None
        self.current_x_data = None
        self.current_y1_data = None
        self.current_y2_data = None
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Add Load Data button at the top
        load_button = QPushButton("Load .mat File")
        load_button.clicked.connect(self.load_data)
        main_layout.addWidget(load_button)
        
        # Create splitter for the two plot areas
        splitter = QSplitter(Qt.Horizontal)
        
        # First plot area
        plot1_widget = QWidget()
        plot1_layout = QVBoxLayout(plot1_widget)
        
        # Create the first matplotlib figure
        self.fig1 = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.ax1.set_xlabel("Input variable")
        self.ax1.set_ylabel("Y1 Axis")
        self.ax2.set_ylabel("Y2 Axis")
        
        self.canvas1 = FigureCanvas(self.fig1)
        plot1_layout.addWidget(self.canvas1)
        
        # Create and add the first toolbar
        self.toolbar1 = CustomNavigationToolbar(self.canvas1, self)
        plot1_layout.addWidget(self.toolbar1)
        
        # Add slider widget to first plot
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        slider_layout.addWidget(spacer)
        
        self.x_value_label = QLabel("X Value:")
        slider_layout.addWidget(self.x_value_label)
        
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(10000)
        self.x_slider.setTickInterval(100)
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.valueChanged.connect(self.update_slider_value)
        self.x_slider.setFixedWidth(100)
        slider_layout.addWidget(self.x_slider)
        
        self.x_value_display = QLabel("0.0")
        self.x_value_display.setMinimumWidth(60)
        slider_layout.addWidget(self.x_value_display)
        
        self.toolbar1.addWidget(slider_widget)
        
        # Second plot area
        plot2_widget = QWidget()
        plot2_layout = QVBoxLayout(plot2_widget)
        
        # Modify the second plot area to include twin axes
        self.fig2 = Figure(figsize=(5, 4), dpi=100)
        self.ax3 = self.fig2.add_subplot(111)
        self.ax4 = self.ax3.twinx()  # Create twin axis
        self.canvas2 = FigureCanvas(self.fig2)
        plot2_layout.addWidget(self.canvas2)
        
        # Create and add the second toolbar
        self.toolbar2 = CustomNavigationToolbar(self.canvas2, self)
        plot2_layout.addWidget(self.toolbar2)
        
        # Add both plot widgets to splitter
        splitter.addWidget(plot1_widget)
        splitter.addWidget(plot2_widget)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        splitter.setSizes([600, 600])  # Equal widths for both plots
        
        # Add controls layout
        controls_layout = QVBoxLayout()
        
        # Output controls 
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Y1:"))
        self.output1_combo = QComboBox()
        self.output1_combo.addItems([""] + ["L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"])
        output_layout.addWidget(self.output1_combo)
        self.output1scale_combo = QComboBox()
        output_layout.addWidget(QLabel("scale:"))
        self.output1scale_combo.addItems(["linear","log"])
        output_layout.addWidget(self.output1scale_combo)

        output_layout.addWidget(QLabel("Y2:"))
        self.output2_combo = QComboBox()
        self.output2_combo.addItems([""] + ["L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"])
        output_layout.addWidget(self.output2_combo)
        output_layout.addWidget(QLabel("scale:"))
        self.output2scale_combo = QComboBox()
        self.output2scale_combo.addItems(["linear","log"])
        output_layout.addWidget(self.output2scale_combo)

        output_layout.addWidget(QLabel("X:"))
        self.inputx_combo = QComboBox()
        self.inputx_combo.addItems([""] + [
            "L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"
        ])
        output_layout.addWidget(self.inputx_combo)
        output_layout.addWidget(QLabel("scale:"))
        self.inputxscale_combo = QComboBox()
        self.inputxscale_combo.addItems(["linear","log"])
        output_layout.addWidget(self.inputxscale_combo)
        
        controls_layout.addLayout(output_layout)

# Input control
        input_layout = QVBoxLayout()

        # Input 1
        input1_layout = QHBoxLayout()
        input1_layout.addWidget(QLabel("Input 1:"))
        self.input1_combo = QComboBox()
        self.input1_combo.addItems([""] + [
            "L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"
        ])
        input1_layout.addWidget(self.input1_combo)
        self.input1_field = QLineEdit()
        input1_layout.addWidget(self.input1_field)
        input_layout.addLayout(input1_layout)

        # Input 2
        input2_layout = QHBoxLayout()
        input2_layout.addWidget(QLabel("Input 2:"))
        self.input2_combo = QComboBox()
        self.input2_combo.addItems([""] + [
            "L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"
        ])
        input2_layout.addWidget(self.input2_combo)
        self.input2_field = QLineEdit()
        input2_layout.addWidget(self.input2_field)
        input_layout.addLayout(input2_layout)

        # Input 3
        input3_layout = QHBoxLayout()
        input3_layout.addWidget(QLabel("Input 3:"))
        self.input3_combo = QComboBox()
        self.input3_combo.addItems([""] + [
            "L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"
        ])
        input3_layout.addWidget(self.input3_combo)
        self.input3_field = QLineEdit()
        input3_layout.addWidget(self.input3_field)
        input_layout.addLayout(input3_layout)

        # Input 4
        input4_layout = QHBoxLayout()
        input4_layout.addWidget(QLabel("Input 4:"))
        self.input4_combo = QComboBox()
        self.input4_combo.addItems([""] + [
            "L", "W", "VGS", "VDS", "VSB", "ID", "VT", "IGD", "IGS", "GM", "GMB", "GDS", "CGG", "CGS",
            "CSG", "CGD", "CDG", "CGB", "CDD", "CSS", "STH", "SFL","CDD_CDG","CDD_CGB","CDD_CGD","CDD_CGG","CDD_CGS","CDD_CSG","CDD_CSS","CDD_GDS","CDD_GM","CDD_GMB","CDD_ID","CDD_IGD","CDD_IGS","CDD_L","CDD_SFL","CDD_STH","CDD_VDS","CDD_VGS","CDD_VSB","CDD_VT","CDD_W","CDG_CDD","CDG_CGB","CDG_CGD","CDG_CGG","CDG_CGS","CDG_CSG","CDG_CSS","CDG_GDS","CDG_GM","CDG_GMB","CDG_ID","CDG_IGD","CDG_IGS","CDG_L","CDG_SFL","CDG_STH","CDG_VDS","CDG_VGS","CDG_VSB","CDG_VT","CDG_W","CGB_CDD","CGB_CDG","CGB_CGD","CGB_CGG","CGB_CGS","CGB_CSG","CGB_CSS","CGB_GDS","CGB_GM","CGB_GMB","CGB_ID","CGB_IGD","CGB_IGS","CGB_L","CGB_SFL","CGB_STH","CGB_VDS","CGB_VGS","CGB_VSB","CGB_VT","CGB_W","CGD_CDD","CGD_CDG","CGD_CGB","CGD_CGG","CGD_CGS","CGD_CSG","CGD_CSS","CGD_GDS","CGD_GM","CGD_GMB","CGD_ID","CGD_IGD","CGD_IGS","CGD_L","CGD_SFL","CGD_STH","CGD_VDS","CGD_VGS","CGD_VSB","CGD_VT","CGD_W","CGG_CDD","CGG_CDG","CGG_CGB","CGG_CGD","CGG_CGS","CGG_CSG","CGG_CSS","CGG_GDS","CGG_GM","CGG_GMB","CGG_ID","CGG_IGD","CGG_IGS","CGG_L","CGG_SFL","CGG_STH","CGG_VDS","CGG_VGS","CGG_VSB","CGG_VT","CGG_W","CGS_CDD","CGS_CDG","CGS_CGB","CGS_CGD","CGS_CGG","CGS_CSG","CGS_CSS","CGS_GDS","CGS_GM","CGS_GMB","CGS_ID","CGS_IGD","CGS_IGS","CGS_L","CGS_SFL","CGS_STH","CGS_VDS","CGS_VGS","CGS_VSB","CGS_VT","CGS_W","CSG_CDD","CSG_CDG","CSG_CGB","CSG_CGD","CSG_CGG","CSG_CGS","CSG_CSS","CSG_GDS","CSG_GM","CSG_GMB","CSG_ID","CSG_IGD","CSG_IGS","CSG_L","CSG_SFL","CSG_STH","CSG_VDS","CSG_VGS","CSG_VSB","CSG_VT","CSG_W","CSS_CDD","CSS_CDG","CSS_CGB","CSS_CGD","CSS_CGG","CSS_CGS","CSS_CSG","CSS_GDS","CSS_GM","CSS_GMB","CSS_ID","CSS_IGD","CSS_IGS","CSS_L","CSS_SFL","CSS_STH","CSS_VDS","CSS_VGS","CSS_VSB","CSS_VT","CSS_W","GDS_CDD","GDS_CDG","GDS_CGB","GDS_CGD","GDS_CGG","GDS_CGS","GDS_CSG","GDS_CSS","GDS_GM","GDS_GMB","GDS_ID","GDS_IGD","GDS_IGS","GDS_L","GDS_SFL","GDS_STH","GDS_VDS","GDS_VGS","GDS_VSB","GDS_VT","GDS_W","GMB_CDD","GMB_CDG","GMB_CGB","GMB_CGD","GMB_CGG","GMB_CGS","GMB_CSG","GMB_CSS","GMB_GDS","GMB_GM","GMB_ID","GMB_IGD","GMB_IGS","GMB_L","GMB_SFL","GMB_STH","GMB_VDS","GMB_VGS","GMB_VSB","GMB_VT","GMB_W","GM_CDD","GM_CDG","GM_CGB","GM_CGD","GM_CGG","GM_CGS","GM_CSG","GM_CSS","GM_GDS","GM_GMB","GM_ID","GM_IGD","GM_IGS","GM_L","GM_SFL","GM_STH","GM_VDS","GM_VGS","GM_VSB","GM_VT","GM_W","ID_CDD","ID_CDG","ID_CGB","ID_CGD","ID_CGG","ID_CGS","ID_CSG","ID_CSS","ID_GDS","ID_GM","ID_GMB","ID_IGD","ID_IGS","ID_L","ID_SFL","ID_STH","ID_VDS","ID_VGS","ID_VSB","ID_VT","ID_W","IGD_CDD","IGD_CDG","IGD_CGB","IGD_CGD","IGD_CGG","IGD_CGS","IGD_CSG","IGD_CSS","IGD_GDS","IGD_GM","IGD_GMB","IGD_ID","IGD_IGS","IGD_L","IGD_SFL","IGD_STH","IGD_VDS","IGD_VGS","IGD_VSB","IGD_VT","IGD_W","IGS_CDD","IGS_CDG","IGS_CGB","IGS_CGD","IGS_CGG","IGS_CGS","IGS_CSG","IGS_CSS","IGS_GDS","IGS_GM","IGS_GMB","IGS_ID","IGS_IGD","IGS_L","IGS_SFL","IGS_STH","IGS_VDS","IGS_VGS","IGS_VSB","IGS_VT","IGS_W","L_CDD","L_CDG","L_CGB","L_CGD","L_CGG","L_CGS","L_CSG","L_CSS","L_GDS","L_GM","L_GMB","L_ID","L_IGD","L_IGS","L_SFL","L_STH","L_VDS","L_VGS","L_VSB","L_VT","L_W","SFL_CDD","SFL_CDG","SFL_CGB","SFL_CGD","SFL_CGG","SFL_CGS","SFL_CSG","SFL_CSS","SFL_GDS","SFL_GM","SFL_GMB","SFL_ID","SFL_IGD","SFL_IGS","SFL_L","SFL_STH","SFL_VDS","SFL_VGS","SFL_VSB","SFL_VT","SFL_W","STH_CDD","STH_CDG","STH_CGB","STH_CGD","STH_CGG","STH_CGS","STH_CSG","STH_CSS","STH_GDS","STH_GM","STH_GMB","STH_ID","STH_IGD","STH_IGS","STH_L","STH_SFL","STH_VDS","STH_VGS","STH_VSB","STH_VT","STH_W","VDS_CDD","VDS_CDG","VDS_CGB","VDS_CGD","VDS_CGG","VDS_CGS","VDS_CSG","VDS_CSS","VDS_GDS","VDS_GM","VDS_GMB","VDS_ID","VDS_IGD","VDS_IGS","VDS_L","VDS_SFL","VDS_STH","VDS_VGS","VDS_VSB","VDS_VT","VDS_W","VGS_CDD","VGS_CDG","VGS_CGB","VGS_CGD","VGS_CGG","VGS_CGS","VGS_CSG","VGS_CSS","VGS_GDS","VGS_GM","VGS_GMB","VGS_ID","VGS_IGD","VGS_IGS","VGS_L","VGS_SFL","VGS_STH","VGS_VDS","VGS_VSB","VGS_VT","VGS_W","VSB_CDD","VSB_CDG","VSB_CGB","VSB_CGD","VSB_CGG","VSB_CGS","VSB_CSG","VSB_CSS","VSB_GDS","VSB_GM","VSB_GMB","VSB_ID","VSB_IGD","VSB_IGS","VSB_L","VSB_SFL","VSB_STH","VSB_VDS","VSB_VGS","VSB_VT","VSB_W","VT_CDD","VT_CDG","VT_CGB","VT_CGD","VT_CGG","VT_CGS","VT_CSG","VT_CSS","VT_GDS","VT_GM","VT_GMB","VT_ID","VT_IGD","VT_IGS","VT_L","VT_SFL","VT_STH","VT_VDS","VT_VGS","VT_VSB","VT_W","W_CDD","W_CDG","W_CGB","W_CGD","W_CGG","W_CGS","W_CSG","W_CSS","W_GDS","W_GM","W_GMB","W_ID","W_IGD","W_IGS","W_L","W_SFL","W_STH","W_VDS","W_VGS","W_VSB","W_VT"
        ])
        input4_layout.addWidget(self.input4_combo)
        self.input4_field = QLineEdit()
        input4_layout.addWidget(self.input4_field)
        input_layout.addLayout(input4_layout)

        controls_layout.addLayout(input_layout)

        # Update buttons
        buttons_layout = QHBoxLayout()
        update_button1 = QPushButton("Update Plot")
        update_button1.clicked.connect(self.update_plot1)
        buttons_layout.addWidget(update_button1)
        
        
        controls_layout.addLayout(buttons_layout)
        
        main_layout.addLayout(controls_layout)
        
        # Enable tooltips
        self.enable_tooltip()
        
        self.setCentralWidget(main_widget)

    def load_data(self):
        """Load .mat file containing transistor data."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load .mat File", "", "MAT files (*.mat)")
        if file_name:
            try:
                data = io.loadmat(file_name)
                print("Available keys in loaded data:", data.keys())

                # Try loading both nch and pch data
                self.nch_data = data.get('nch', None)
                self.pch_data = data.get('pch', None)

                if self.nch_data is None and self.pch_data is None:
                    raise ValueError("Neither 'nch' nor 'pch' data found in the .mat file.")
                QMessageBox.information(self, "Data Loaded", "Data loaded successfully!")
            except Exception as e:
                print(f"Error loading data: {e}")
                QMessageBox.critical(self, "Load Error", f"Could not load .mat file: {str(e)}")
           
    def update_slider_value(self):
        """Update the display when slider value changes and update both plots"""
        if self.current_x_data is not None:
            # Convert slider value to actual x-axis value
            slider_pos = self.x_slider.value() / 10000
            x_min, x_max = np.min(self.current_x_data), np.max(self.current_x_data)
            x_value = x_min + slider_pos * (x_max - x_min)
            
            # Update label
            self.x_value_display.setText(f"{x_value:.6f}")
            
            # Remove old vertical lines from first plot
            for line in self.ax1.lines:
                if hasattr(line, 'get_label') and line.get_label() == 'vline':
                    line.remove()
            
            # Draw new vertical line on first plot
            self.ax1.axvline(x=x_value, linestyle='--', color='black', label='vline')
            self.canvas1.draw()
            
            # Update second plot
            self.update_intersection_plot(x_value)
    
    def enable_tooltip(self):
        """
        Enable tooltips to display x and y values when hovering over the plot.
        """
        def on_hover(event):
            if event.inaxes == self.ax1 or event.inaxes == self.ax2:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    self.statusBar().showMessage(f"x: {x:.2f}, y: {y:.2f}")
            else:
                self.statusBar().clearMessage()

        self.canvas1.mpl_connect("motion_notify_event", on_hover)
        
    def prepare_lookup1(self):
        """Prepare and call lookup function with selected inputs."""
        try:
            x_var = self.inputx_combo.currentText()
            y1_var = self.output1_combo.currentText()
            y2_var = self.output2_combo.currentText()
            x_scale_var=self.inputxscale_combo.currentText()
            y1_scale_var=self.output1scale_combo.currentText()
            y2_scale_var=self.output2scale_combo.currentText()
            # Print detailed debug information
            print("------- Lookup Debug -------")
            print(f"X Variable: {x_var}")
            print(f"Y1 Variable: {y1_var}")
            print(f"Y2 Variable: {y2_var}")
        
            # Check if data is loaded
            if self.nch_data is None and self.pch_data is None:
                print("ERROR: No data loaded!")
                QMessageBox.critical(self, "Data Error", "No .mat file has been loaded")
                return None
            
            # Store which parameter has multiple values and what those values are
            self.varying_param = None
            self.varying_values = None
            
            
            # Prepare input parameters
            input_params = {}
            input_combos = [
                (self.input1_combo, self.input1_field),
                (self.input2_combo, self.input2_field),
                (self.input3_combo, self.input3_field),
                (self.input4_combo, self.input4_field)
            ]

            for combo, field in input_combos:
                param = combo.currentText()
                value = field.text()
                if param and value:
                    try:
                        # Try different parsing strategies
                        if ':' in value:
                        # Parse colon-separated range
                            start, step, end = map(float, value.split(':'))
                            parsed_value = np.arange(start, end, step)
                        elif ',' in value:
                            # Comma-separated list
                            parsed_value = np.array([float(x.strip()) for x in value.split(',')])
                        else:
                            # Single value
                            parsed_value = float(value)
                    
                        input_params[param] = parsed_value
                        print(f"Input Parameter: {param} = {parsed_value}")
                    except ValueError as ve:
                        print(f"ERROR parsing {param}: {ve}")
                        QMessageBox.warning(self, "Input Error", f"Invalid value for {param}: {value}")
                        return None
            
            for combo, field in input_combos:
                param = combo.currentText()
                value = field.text()
                if param and value:
                    if ':' in value or ',' in value:
                        self.varying_param = param
                        if ':' in value:
                            start, step, end = map(float, value.split(':'))
                            self.varying_values = np.arange(start, end, step)
                        else:
                            self.varying_values = np.array([float(x.strip()) for x in value.split(',')])            
            
            # Initialize results
            x_result = None
            y1_result = None
            y2_result = None
            # Attempt lookup for each output variable
            try:
                print("\nAttempting Lookup:")
                if self.nch_data is not None:
                    if y1_var != "": y1_result = lookup(self.nch_data, y1_var, **input_params)
                    if y2_var != "":y2_result = lookup(self.nch_data, y2_var, **input_params)
                    x_result = lookup(self.nch_data,x_var,**input_params)
                elif self.pch_data is not None:
                    if y1_var != "":y1_result = lookup(self.pch_data, y1_var, **input_params)
                    if y2_var != "":y2_result = lookup(self.pch_data, y2_var, **input_params)
                    x_result = lookup(self.pch_data,x_var,**input_params)
                else:
                    print("Invalid Data loaded")
                # Print detailed lookup results
                if y1_result is not None:
                    print("Y1 Result:")
                    print(y1_result)
                if y2_result is not None:
                    print("\nY2 Result:")
                    print(y2_result)

                print("\nX Result:")
                print(x_result)

                # Call plot_array with results
                if x_result is not None:
                    self.current_x_data = x_result  # Store the x-axis data                    
                    if y1_result is not None and y2_result is not None:
                        plot_array(x_result,y1_result, y2_result,canvas=self.canvas1,ax1=self.ax1,ax2=self.ax2,x_label=x_var,y1_label=y1_var,y2_label=y2_var,x_scale=x_scale_var,y1_scale=y1_scale_var,y2_scale=y2_scale_var)
                    elif y1_result is not None and y2_result is None:
                        plot_array(x_result,y1_result,canvas=self.canvas1,ax1=self.ax1,ax2=self.ax2,x_label=x_var,y1_label=y1_var,y2_label=y2_var,x_scale=x_scale_var,y1_scale=y1_scale_var,y2_scale=y2_scale_var)
                    elif y1_result is None and y2_result is not None:
                        plot_array(x_result,y2_result,canvas=self.canvas1,ax1=self.ax1,ax2=self.ax2,x_label=x_var,y1_label=y1_var,y2_label=y2_var,x_scale=x_scale_var,y1_scale=y1_scale_var,y2_scale=y2_scale_var)
                
                    # Reset slider to 0 and update the vertical line
                    self.x_slider.setValue(0)
                    self.update_slider_value()
                else:
                    QMessageBox.warning(self, "Lookup Error", "Failed to retrieve data for plotting")

            except Exception as lookup_error:
                print(f"Lookup Error: {lookup_error}")
                QMessageBox.critical(self, "Lookup Error", str(lookup_error))
    
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")

    

    def update_plot1(self):
        """Update the first plot and store current data"""
        if self.nch_data is None and self.pch_data is None:
            QMessageBox.warning(self, "Data Error", "Please load a .mat file first")
            return
        
        result = self.prepare_lookup1()
        if result is not None:
            self.current_x_data, self.current_y1_data, self.current_y2_data = result
            
            # Reset slider to 0 and update both plots
            self.x_slider.setValue(0)
            self.update_slider_value()
    
    def update_plot2(self):
        """Update the second plot with current intersection points"""
        if self.current_x_data is None:
            QMessageBox.warning(self, "Plot Error", "Please update Plot 1 first")
            return
        
        # Get current slider value and update intersection plot
        slider_pos = self.x_slider.value() / 10000
        x_min, x_max = np.min(self.current_x_data), np.max(self.current_x_data)
        x_value = x_min + slider_pos * (x_max - x_min)
        self.update_intersection_plot(x_value)

    def update_intersection_plot(self, x_value):
        """Update the second plot with values across the varying parameter and find intersection points."""
        print("Debugging intersection plot:")
        print(f"Varying param: {self.varying_param}")
        print(f"Varying values: {self.varying_values}")
        print(f"Current x value: {x_value}")

        try:
            x_var = self.inputx_combo.currentText()
            y1_var = self.output1_combo.currentText()
            y2_var = self.output2_combo.currentText()

            # Check if data is loaded
            if self.nch_data is None and self.pch_data is None:
                print("ERROR: No data loaded!")
                QMessageBox.critical(self, "Data Error", "No .mat file has been loaded")
                return

            # Prepare input parameters
            input_params = {}
            input_combos = [
                (self.input1_combo, self.input1_field),
                (self.input2_combo, self.input2_field),
                (self.input3_combo, self.input3_field),
                (self.input4_combo, self.input4_field),
            ]

            for combo, field in input_combos:
                param = combo.currentText()
                if param:  # Ensure the parameter is selected
                    if isinstance(field, QLineEdit):
                        value = field.text()
                    else:
                        value = field  # Handle non-text fields

                    if value:
                        try:
                            # Parse value for different formats
                            if ':' in value:
                                start, step, end = map(float, value.split(':'))
                                parsed_value = np.arange(start, end, step)
                            elif ',' in value:
                                parsed_value = np.array([float(x.strip()) for x in value.split(',')])
                            else:
                                parsed_value = float(value)

                            input_params[param] = parsed_value
                            print(f"Input Parameter: {param} = {parsed_value}")
                        except ValueError as ve:
                            print(f"ERROR parsing {param}: {ve}")
                            QMessageBox.warning(self, "Input Error", f"Invalid value for {param}: {value}")
                            return

            try:
                print("\nAttempting Lookup:")
                if self.nch_data is not None:
                    if y1_var != "":self.current_y1_data = lookup(self.nch_data, y1_var, x_var, x_value, **input_params)
                    if y2_var != "":self.current_y2_data = lookup(self.nch_data, y2_var, x_var, x_value, **input_params)
                elif self.pch_data is not None:
                    if y1_var != "":self.current_y1_data = lookup(self.pch_data, y1_var, x_var, x_value, **input_params)
                    if y2_var != "":self.current_y2_data = lookup(self.pch_data, y2_var, x_var, x_value, **input_params)
                else:
                    print("Invalid Data loaded")

            except Exception as lookup_error:
                print(f"Lookup Error: {lookup_error}")
                QMessageBox.critical(self, "Lookup Error", str(lookup_error))
                return

        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {str(e)}")
            return

        if self.varying_param is None or self.varying_values is None:
            print("No varying parameter found")
            return

        # Clear both axes
        self.ax3.clear()
        self.ax4.clear()

        # Flatten y1_values and y2_values if necessary
        y1_values = np.ravel(self.current_y1_data) if self.current_y1_data is not None else None
        y2_values = np.ravel(self.current_y2_data) if self.current_y2_data is not None else None

        # Debug print for shapes
        print(f"self.varying_values shape: {self.varying_values.shape}")
        print(f"y1_values shape: {y1_values.shape if y1_values is not None else 'None'}")
        print(f"y2_values shape: {y2_values.shape if y2_values is not None else 'None'}")

        # Plot Y1 on left axis
        if y1_values is not None:
            self.ax3.plot(self.varying_values, y1_values, 'r-', label=y1_var)
            self.ax3.set_ylabel(y1_var, color='r')
            self.ax3.tick_params(axis='y', labelcolor='r')
    
        # Plot Y2 on right axis
        if y2_values is not None:
            self.ax4.plot(self.varying_values, y2_values, 'b-', label=y2_var)
            self.ax4.set_ylabel(y2_var, color='b')
            self.ax4.tick_params(axis='y', labelcolor='b')

        # Set x-axis label
        self.ax3.set_xlabel(self.varying_param)

        # Add a title showing the x-value
        self.ax3.set_title(f'Values at {self.inputx_combo.currentText()} = {x_value:.3f}')

        # Add grid (only for left axis to avoid cluttering)
        self.ax3.grid(True, alpha=0.3)

        # Draw the canvas
        self.canvas2.draw()
    
class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

        # Add "Zoom Out" button
        self.add_zoom_out_button()

    def add_zoom_out_button(self):
        """
        Add a "Zoom Out" button to the toolbar.
        """
        # Create an action for Zoom Out
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setToolTip("Zoom out to see a larger view")
        zoom_out_action.triggered.connect(self.zoom_out)
        self.addAction(zoom_out_action)

    def zoom_out(self):
        """
        Reset the view to a zoomed-out state for both x-axis and y-axes.
        """
        for ax in self.canvas.figure.axes:
            # Get current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Calculate zoomed-out limits (increase range by 25%)
            x_range = (xlim[1] - xlim[0]) * 1.25
            y_range = (ylim[1] - ylim[0]) * 1.25

            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            # Set new limits for x-axis and y-axis
            ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
            ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])

        # Redraw the canvas
        self.canvas.draw()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
