import sys
import numpy as np
import pyedflib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QScrollArea, QSlider, 
    QHBoxLayout, QSpinBox, QLabel, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QEvent
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

        # Добавляем новые переменные для выделения области
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.selection_rect = None
        self.is_adding_selection = False
        self.selection_rects = []  # Список для хранения всех выделений

        self.ctrl_pressed = False
        
        # Подключаем обработчики событий мыши
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Добавляем текстовые поля для ввода времени
        self.selection_controls = QHBoxLayout()
        
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setPlaceholderText("HH:MM:SS")
        self.start_time_edit.setMaximumWidth(100)
        
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("HH:MM:SS")
        self.end_time_edit.setMaximumWidth(100)
        
        self.btn_set_selection = QPushButton("Set Selection")
        self.btn_set_selection.setMaximumWidth(100)
        self.btn_set_selection.clicked.connect(self.set_selection_from_text)
        
        self.selection_controls.addWidget(QLabel("From:"))
        self.selection_controls.addWidget(self.start_time_edit)
        self.selection_controls.addWidget(QLabel("To:"))
        self.selection_controls.addWidget(self.end_time_edit)
        self.selection_controls.addWidget(self.btn_set_selection)
        
        # Добавляем на панель управления
        self.control_layout.insertLayout(1, self.selection_controls)
        
        # Кнопка для запуска сегментации
        self.btn_segment = QPushButton("Segment Selected")
        self.btn_segment.clicked.connect(self.segment_selected)
        self.btn_segment.setEnabled(False)
        self.control_layout.addWidget(self.btn_segment)

    def eventFilter(self, obj, event):
        """Перехватываем события клавиатуры для всего приложения"""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Control:
                self.ctrl_pressed = True
        elif event.type() == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_Control:
                self.ctrl_pressed = False
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        """Устанавливаем фильтр событий при показе окна"""
        QApplication.instance().installEventFilter(self)
        super().showEvent(event)

    def closeEvent(self, event):
        """Удаляем фильтр событий при закрытии окна"""
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.is_selecting = True
        self.selection_start = event.xdata
        
        # Удаляем все выделения только если Ctrl не нажат
        if not self.ctrl_pressed:
            self.clear_all_selections()
    
    def on_motion(self, event):
        if not self.is_selecting or event.inaxes != self.ax:
            return
        
        # Очищаем предыдущий прямоугольник выделения
        if self.selection_rect:
            self.selection_rect.remove()
        
        # Рисуем новый прямоугольник выделения
        self.selection_rect = self.ax.axvspan(
            self.selection_start, event.xdata,
            facecolor='yellow', alpha=0.3,
            edgecolor='orange', linewidth=1
        )
        self.canvas.draw_idle()

    def parse_time_input(self, time_str):
        """Конвертирует строку времени в секунды"""
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except:
            return None

    def set_selection_from_text(self):
        """Устанавливает выделение из текстовых полей"""

        self.clear_all_selections()

        start_time = self.parse_time_input(self.start_time_edit.text())
        end_time = self.parse_time_input(self.end_time_edit.text())
        
        if start_time is None or end_time is None:
            return
            
        # Проверяем границы
        max_time = len(next(iter(self.signals.values()))) / self.sample_rates[0]
        start_time = max(0, min(start_time, max_time))
        end_time = max(0, min(end_time, max_time))
        
        if start_time >= end_time:
            return
            
        # Устанавливаем выделение
        self.selection_start = start_time
        self.selection_end = end_time
        
        # Обновляем отображение
        self.update_selection_rect()
        self.btn_segment.setEnabled(True)

    def clear_all_selections(self):
        """Удаляет все выделения"""
        for rect in self.selection_rects:
            try:
                # Проверяем, существует ли прямоугольник на осях
                if rect in self.ax.patches:
                    rect.remove()
            except:
                continue
        self.selection_rects = []
        self.canvas.draw_idle()

    def update_selection_rect(self):
        """Создает новое выделение и добавляет его в список"""
        if self.selection_start is not None and self.selection_end is not None:
            rect = self.ax.axvspan(
                self.selection_start, self.selection_end,
                facecolor='yellow', alpha=0.3,
                edgecolor='orange', linewidth=1
            )
            self.selection_rects.append(rect)  # Добавляем в список
            self.canvas.draw_idle()

    
    def on_release(self, event):
        if not self.is_selecting or event.inaxes != self.ax:
            return
        
        self.is_selecting = False
        self.selection_end = event.xdata
        
        # Убедимся, что start < end
        if self.selection_start > self.selection_end:
            self.selection_start, self.selection_end = self.selection_end, self.selection_start

        # Активируем кнопку сегментации
        self.btn_segment.setEnabled(True)

        # Обновляем текстовые поля
        self.start_time_edit.setText(self.format_time(self.selection_start))
        self.end_time_edit.setText(self.format_time(self.selection_end))
        
        self.update_selection_rect()
    
    def segment_selected(self):
        if not self.signals or self.selection_start is None or self.selection_end is None:
            return
        
        # Получаем индексы для выделенной области
        fs = self.sample_rates[0]
        start_idx = int(self.selection_start * fs)
        end_idx = int(self.selection_end * fs)

        if end_idx - start_idx < 4000:
            QMessageBox.warning(self, "Segment Too Short", "The selected segment is too short. Please select a segment with at least 25 seconds.")
            return
        
        # Берем только выделенные данные
        selected_signals = {
            label: signal[start_idx:end_idx] 
            for label, signal in self.signals.items()
        }
        
        # Выполняем сегментацию только для выделенной области
        signal_copy = list(selected_signals.values()).copy()
        signal_copy.reverse()
        #print("Shape", np.array(signal_copy).shape)
        selected_mask = SendPredictsGUI(np.array(signal_copy), self.sample_rates[0])
        
        # Обновляем полную маску (только выделенный участок)
        full_mask_length = len(next(iter(self.signals.values())))
        if self.mask is None or len(self.mask) != full_mask_length:
            self.mask = np.zeros(full_mask_length)
        
        self.mask[start_idx:start_idx + len(selected_mask)] = selected_mask
        
        # Обновляем отображение
        self.update_plot()
        self.btn_segment.setEnabled(False)
        
        # Убираем выделение
        self.clear_all_selections()
        self.canvas.draw_idle()

        # Очищаем текстовые поля после сегментации
        self.start_time_edit.clear()
        self.end_time_edit.clear()
    
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
            #print("Right shape", np.array(signal_copy).shape)
            #self.mask = SendPredictsGUI(np.array(signal_copy), self.sample_rates[0])
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
        if self.mask is not None:
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

        # Добавляем отображение текущего выделения
        if self.selection_rect and self.selection_start and self.selection_end:
            self.selection_rect = self.ax.axvspan(
                self.selection_start, self.selection_end,
                facecolor='yellow', alpha=0.3,
                edgecolor='orange', linewidth=1
            )
        self.update_selection_rect()
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UltraCompactEEGViewer()
    viewer.show()
    sys.exit(app.exec())