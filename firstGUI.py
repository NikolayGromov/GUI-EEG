import sys
import numpy as np
import pyedflib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QScrollArea, QSlider, 
    QHBoxLayout, QSpinBox, QLabel
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

from network import *


class UltraCompactEEGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer - Ultra Compact Mode")
        self.setGeometry(100, 100, 1200, 800)
        
        # Основные параметры
        self.edf_file = None
        self.signals = {}
        self.signal_labels = []
        self.sample_rates = []
        self.time_range = (0, 10)
        self.current_position = 0
        self.mask = None
        
        # Настройка интерфейса
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Минималистичная панель управления
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout()
        self.control_layout.setContentsMargins(5, 2, 5, 2)
        
        self.btn_load = QPushButton("Load")
        self.btn_load.setMaximumWidth(60)
        self.btn_load.clicked.connect(self.load_edf)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.valueChanged.connect(self.update_plot)
        
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(1, 600)
        self.zoom_spin.setValue(10)
        self.zoom_spin.setMaximumWidth(60)
        self.zoom_spin.valueChanged.connect(self.update_time_range)
        
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addWidget(QLabel("Window:"))
        self.control_layout.addWidget(self.zoom_spin)
        self.control_layout.addWidget(QLabel("sec"))
        self.control_layout.addWidget(self.time_slider)
        self.control_panel.setLayout(self.control_layout)
        self.control_panel.setMaximumHeight(30)
        
        # Область графика
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.figure.subplots_adjust(
            left=0.01, right=0.99,
            top=0.99, bottom=0.01,
            hspace=0, wspace=0
        )
        self.canvas = FigureCanvas(self.figure)
        
        # Компоновка
        self.main_layout.addWidget(self.control_panel)
        self.main_layout.addWidget(self.canvas)
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        
        # Инициализация осей
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()
    
    def setup_axes(self):
        """Полностью убираем все внешние элементы осей"""
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.xaxis.set_major_formatter(
            FuncFormatter(self.format_time_axis))
        
        # Сетка (только горизонтальные линии)
        self.ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Убираем рамку
        for spine in self.ax.spines.values():
            spine.set_visible(False)
    
    def format_time_axis(self, secs, pos=None):
        """Форматирует секунды в ЧЧ:ММ:СС."""
        td = timedelta(seconds=secs)
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def load_edf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf)"
        )
        
        if not file_path:
            return
            
        try:
            self.edf_file = pyedflib.EdfReader(file_path)
            self.signal_labels = self.edf_file.getSignalLabels()[::-1]  # Обратный порядок
            self.sample_rates = [self.edf_file.getSampleFrequency(i) for i in range(len(self.signal_labels))]
            
            # Загрузка данных
            self.signals = {
                label: self.edf_file.readSignal(len(self.signal_labels)-1-i)
                for i, label in enumerate(self.signal_labels)
            }
            
            max_len = len(next(iter(self.signals.values())))
            max_time = max_len / self.sample_rates[0]
            self.time_slider.setRange(0, int(max_time - self.time_range[1]))
            signal_copy = list(self.signals.values()).copy()
            signal_copy.reverse()
            self.mask = SendPredictsGUI(np.array(signal_copy), self.sample_rates[0])
            self.update_plot()
            self.btn_load.setText("✓ Loaded")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def update_time_range(self):
        self.time_range = (self.current_position, self.current_position + self.zoom_spin.value())
        self.update_plot()

    def format_time(self, seconds):
        """Форматирует время для встроенных меток"""
        return f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    
    def update_plot(self, position=None):
        if position is not None:
            self.current_position = position
            self.time_range = (position, position + self.zoom_spin.value())
        
        if not self.signals:
            return
            
        self.ax.clear()
        self.setup_axes()
        
        # Рисуем все каналы
        for i, (label, signal) in enumerate(self.signals.items()):
            fs = self.sample_rates[i]
            start = int(self.time_range[0] * fs)
            end = int(self.time_range[1] * fs)
            time = np.linspace(self.time_range[0], self.time_range[1], end-start)
            
            # Нормализация и отрисовка
            normalized = (signal[start:end] - np.mean(signal)) / (2 * np.std(signal))# изменять по кнопке коэф
            self.ax.plot(time, normalized + i, color='#1f77b4', linewidth=0.7)
            
            # Метка канала (внутри графика)
            self.ax.text(
                self.time_range[0] + 0.02 * (self.time_range[1]-self.time_range[0]), 
                i + 0.1, label,
                fontsize=8, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
            )
        # отрисовка маски
        changes = np.diff(self.mask[start:end], prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e > s and e < len(time):  # Игнорируем пустые интервалы
                self.ax.fill_betweenx(
                    y=[-0.5, len(self.signals)-0.5],  # Заливка по всем каналам
                    x1=time[s],
                    x2=time[e],
                    facecolor='red',
                    alpha=0.3,
                    edgecolor='none'
                )

        margin = (self.time_range[1] - self.time_range[0]) * 0.02
        for t in range(int(self.time_range[0]), int(self.time_range[1]) + 1, 5):
            if t >= self.time_range[0] and t <= self.time_range[1]:
                x_pos = t
                if t - self.time_range[0] < margin:
                    x_pos = self.time_range[0] + margin  # Отодвигаем от левого края
                elif self.time_range[1] - t < margin:
                    x_pos = self.time_range[1] - margin  # Отодвигаем от правого края
                self.ax.text(
                    x_pos, -0.2, self.format_time(t),
                    fontsize=8, ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, pad=2))
            

        # Вертикальные линии сетки (каждую секунду)
        for t in range(int(self.time_range[0]), int(self.time_range[1]) + 1):
            if t >= self.time_range[0] and t <= self.time_range[1]:
                self.ax.axvline(t, color='black', linestyle=':', alpha=1, linewidth=0.5)
        
        # Горизонтальные линии между каналами
        for i in range(len(self.signals)):
            self.ax.axhline(i, color='black', linestyle='-', alpha=1, linewidth=0.5)
        
        # Границы отображения
        self.ax.set_xlim(self.time_range)
        self.ax.set_ylim(-0.5, len(self.signals)-0.5)
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UltraCompactEEGViewer()
    viewer.show()
    sys.exit(app.exec())