import sys
import numpy as np
import pyedflib
import mne
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QScrollArea, QSlider, 
    QHBoxLayout, QSpinBox, QLabel, QLineEdit, QMessageBox, QDockWidget, QListWidget, QListWidgetItem
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
        self.file_path = None
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
        
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_edf)
        self.control_layout.addWidget(self.btn_save)
        
        
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
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

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

        self.normalization_factor = 2.0  # Дефолтное значение
    
        # Создаем контейнерный виджет для элементов управления нормализацией
        self.norm_control_widget = QWidget()
        self.norm_control_layout = QHBoxLayout(self.norm_control_widget)
        self.norm_control_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_norm_decrease = QPushButton("-")
        self.btn_norm_decrease.setMaximumWidth(30)
        self.btn_norm_decrease.clicked.connect(self.decrease_normalization)
        
        self.norm_label = QLabel(f"Norm: {self.normalization_factor}")
        self.norm_label.setMinimumWidth(60)
        
        self.btn_norm_increase = QPushButton("+")
        self.btn_norm_increase.setMaximumWidth(30)
        self.btn_norm_increase.clicked.connect(self.increase_normalization)
        
        self.norm_control_layout.addWidget(self.btn_norm_decrease)
        self.norm_control_layout.addWidget(self.norm_label)
        self.norm_control_layout.addWidget(self.btn_norm_increase)
        
        # Добавляем виджет на панель управления (например, после zoom_spin)
        self.control_layout.insertWidget(4, self.norm_control_widget)  # Подберите подходящую позицию
        
        # Кнопка для запуска сегментации
        self.btn_segment = QPushButton("Segment Selected")
        self.btn_segment.clicked.connect(self.segment_selected)
        self.btn_segment.setEnabled(False)
        self.control_layout.addWidget(self.btn_segment)

        # Панель аннотаций (выдвижная)
        self.annotation_dock = QDockWidget("Аннотации", self)
        self.annotation_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | 
                                    QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.annotation_widget = QWidget()
        self.annotation_layout = QVBoxLayout()
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemDoubleClicked.connect(self.go_to_annotation)
        
        self.annotation_input = QLineEdit()
        self.annotation_input.setPlaceholderText("Текст аннотации")
        
        self.btn_add_annotation = QPushButton("Добавить аннотацию")
        self.btn_add_annotation.clicked.connect(self.add_annotation)
        
        self.annotation_layout.addWidget(QLabel("Список аннотаций:"))
        self.annotation_layout.addWidget(self.annotation_list)
        self.annotation_layout.addWidget(self.annotation_input)
        self.annotation_layout.addWidget(self.btn_add_annotation)

        self.btn_remove_annotation = QPushButton("Удалить аннотацию")
        self.btn_remove_annotation.clicked.connect(self.remove_selected_annotation)
        self.annotation_layout.addWidget(self.btn_remove_annotation)
        
        self.annotation_widget.setLayout(self.annotation_layout)
        self.annotation_dock.setWidget(self.annotation_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.annotation_dock)

        # Для работы с аннотациями EDF
        self.edf_annotations = []

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
        
        # # Очищаем предыдущий прямоугольник выделения
        if self.selection_rect:
            self.selection_rect.remove()

        # if not self.ctrl_pressed:
        #     self.clear_all_selections()
        
        # Рисуем новый прямоугольник выделения
        self.selection_rect = self.ax.axvspan(
            self.selection_start, event.xdata,
            facecolor='yellow', alpha=0.3,
            edgecolor='orange', linewidth=1
        )
        self.canvas.draw_idle()

    def on_scroll(self, event):
        """Обработчик прокрутки колесика мыши"""
        if not self.signals:
            return
        self.selection_rect = None
        # Определяем направление прокрутки
        scroll_step = 1  # Шаг прокрутки в секундах
        if event.button == 'up':
            # Прокрутка вверх (влево)
            new_position = max(0, self.current_position - scroll_step)
        elif event.button == 'down':
            # Прокрутка вниз (вправо)
            max_position = len(next(iter(self.signals.values()))) / self.sample_rates[0] - self.zoom_spin.value()
            new_position = min(max_position, self.current_position + scroll_step)
        else:
            return
        
        # Обновляем позицию
        self.current_position = new_position
        self.time_slider.setValue(int(new_position))
        self.update_plot()

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
        #self.canvas.draw_idle()
        self.update_plot()

    def update_selection_rect(self):
        """Создает новое выделение и добавляет его в список"""
        print(len(self.selection_rects))
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
        # Находим все сегменты в маске и добавляем их как аннотации
        changes = np.diff(selected_mask, prepend=0, append=0)
        segment_starts = np.where(changes == 1)[0]
        segment_ends = np.where(changes == -1)[0]
        
        # Инициализируем аннотации, если они еще не созданы
        if len(self.edf_annotations) == 0:
            self.edf_annotations = [np.array([]), np.array([]), np.array([])]
        
        # Добавляем каждую найденную область как аннотацию
        for s, e in zip(segment_starts, segment_ends):
            if e > s:  # Игнорируем пустые интервалы
                start_time = (start_idx + s) / fs
                duration = (e - s) / fs
                self.edf_annotations[0] = np.append(self.edf_annotations[0], start_time)
                self.edf_annotations[1] = np.append(self.edf_annotations[1], duration)
                self.edf_annotations[2] = np.append(self.edf_annotations[2], "Network Prediction")
    
        # Обновляем список аннотаций
        self.update_annotation_list()
        
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
        self.file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf)"
        )
        
        if not self.file_path:
            return
            
        try:
            self.edf_file = pyedflib.EdfReader(self.file_path)
            self.signal_labels = self.edf_file.getSignalLabels()[::-1]  # Обратный порядок
            self.sample_rates = [self.edf_file.getSampleFrequency(i) for i in range(len(self.signal_labels))]
            
            # Загрузка данных
            self.signals = {
                label: self.edf_file.readSignal(len(self.signal_labels)-1-i)
                for i, label in enumerate(self.signal_labels)
            }
            
            # Загрузка аннотаций
            self.edf_annotations = list(self.edf_file.readAnnotations())
            self.update_annotation_list()
            max_len = len(next(iter(self.signals.values())))
            max_time = max_len / self.sample_rates[0]
            self.time_slider.setRange(0, int(max_time - self.time_range[1]))
            self.update_plot()
            self.btn_load.setText("✓ Loaded")
            
        except Exception as e:
            print(f"Error: {e}")

    def save_edf(self):
        if not self.edf_file:
            QMessageBox.warning(self, "Ошибка", "Нет открытого EDF файла")
            return
        
        save_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save EDF File", "", "EDF Files (*.edf)"
        )
        
        if not save_file_path:
            return
        
        try:
            data = mne.io.read_raw_edf(self.file_path)
            # Создаем временный raw объект из текущих данных
            
            # Добавляем аннотации если они есть
            if hasattr(self, 'edf_annotations') and len(self.edf_annotations) == 3:
                onsets = self.edf_annotations[0]
                durations = np.clip(self.edf_annotations[1], a_min=0, a_max=None)
                descriptions = self.edf_annotations[2]
                
                
                annotations = mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions
                )
                data.set_annotations(annotations)
            
            # Сохраняем файл
            data.export(save_file_path, fmt='edf', overwrite=True)
            QMessageBox.information(self, "Успех", "Файл успешно сохранен")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")
            print(f"Ошибка сохранения: {e}")
    
    def update_time_range(self):
        self.time_range = (self.current_position, self.current_position + self.zoom_spin.value())
        self.update_plot()

    def format_time(self, seconds):
        """Форматирует время для встроенных меток"""
        return f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    
    # Нормализация
    
    def increase_normalization(self):
        """Увеличивает коэффициент нормализации на 0.5"""
        self.normalization_factor += 0.5
        self.update_normalization_display()
        self.update_plot()

    def decrease_normalization(self):
        """Уменьшает коэффициент нормализации на 0.5 (но не менее 0.5)"""
        self.normalization_factor = max(0.5, self.normalization_factor - 0.5)
        self.update_normalization_display()
        self.update_plot()

    def update_normalization_display(self):
        """Обновляет отображение текущего коэффициента"""
        self.norm_label.setText(f"Norm: {self.normalization_factor:.1f}")

    # Аннотации

    def update_annotation_list(self):
        self.annotation_list.clear()
        for i in range(len(self.edf_annotations[0])):
            start = float(self.edf_annotations[0][i])
            duration = float(self.edf_annotations[1][i])
            text = self.edf_annotations[2][i]
            item = QListWidgetItem(f"{self.format_time(start)}: {text}")
            if duration > 0:
                item = QListWidgetItem(f"{self.format_time(start)} - {self.format_time(start+duration)}: {text}")
            item.setData(Qt.ItemDataRole.UserRole, (start, duration, text))
            self.annotation_list.addItem(item)

    def add_annotation(self):
        """Добавляет аннотации ко всем выделенным областям"""
        if not self.selection_rects:
            QMessageBox.warning(self, "Ошибка", "Нет выделенных областей")
            return
        
        annotation_text = self.annotation_input.text().strip()
        if not annotation_text:
            QMessageBox.warning(self, "Ошибка", "Введите текст аннотации")
            return
        
        # Для каждого выделенного прямоугольника добавляем аннотацию
        uniq_annots = set()
        for rect in self.selection_rects:
            try:
                # Получаем координаты прямоугольника
                start = rect.get_x()  # Минимальная X-координата
                duration = rect.get_width() 
                if (start, duration) not in uniq_annots:
                    uniq_annots.add((start, duration))
                    # Добавляем аннотацию в правильном формате
                    self.edf_annotations[0] = np.append(self.edf_annotations[0], start)
                    self.edf_annotations[1] = np.append(self.edf_annotations[1], duration)
                    self.edf_annotations[2] = np.append(self.edf_annotations[2], annotation_text)
            except Exception as e:
                print(f"Ошибка добавления аннотации: {e}")
                continue
        
        # Обновляем интерфейс
        self.update_annotation_list()
        self.annotation_input.clear()
        self.update_plot()
        QApplication.processEvents()

    def remove_selected_annotation(self):
        """Удаляет выбранную аннотацию из списка"""
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Выберите аннотацию для удаления")
            return
        
        # Получаем индексы выбранных элементов (в обратном порядке, чтобы удаление не сбивало индексы)
        selected_indices = sorted([self.annotation_list.row(item) for item in selected_items], reverse=True)
        
        # Удаляем аннотации из данных
        for idx in selected_indices:
            if len(self.edf_annotations[0]) > idx:
                self.edf_annotations[0] = np.delete(self.edf_annotations[0], idx)
                self.edf_annotations[1] = np.delete(self.edf_annotations[1], idx)
                self.edf_annotations[2] = np.delete(self.edf_annotations[2], idx)
        
        # Обновляем список и график
        self.update_annotation_list()
        self.update_plot()

    def go_to_annotation(self, item):
        """Переходит к выбранной аннотации"""
        start, duration, _ = item.data(Qt.ItemDataRole.UserRole)
        self.time_range = (max(0, start - 5), start + 5)  # Показываем 10 секунд вокруг аннотации
        self.current_position = max(0, start - 5)
        self.time_slider.setValue(int(max(0, start - 5)))
        self.update_plot()
    
    def update_plot(self, position=None):
        self.selection_rect = None

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
            normalized = (signal[start:end] - np.mean(signal)) / (self.normalization_factor * np.std(signal))# изменять по кнопке коэф
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
        # Восстанавливаем все выделения из selection_rects
        for rect in self.selection_rects:
            try:
                # Получаем координаты прямоугольника
                start = rect.get_x()  # начальная координата X
                end = start + rect.get_width()  # конечная координата X
                
                # Проверяем, попадает ли выделение в текущий видимый диапазон
                if end > self.time_range[0] and start < self.time_range[1]:
                    visible_start = max(start, self.time_range[0])
                    visible_end = min(end, self.time_range[1])
                    
                    new_rect = self.ax.axvspan(
                        visible_start, visible_end,
                        facecolor='yellow', alpha=0.3,
                        edgecolor='orange', linewidth=1
                    )
                    # Сохраняем ссылку на новый прямоугольник
                    #self.selection_rects[self.selection_rects.index(rect)] = new_rect
            except Exception as e:
                print(f"Ошибка восстановления выделения: {e}")
                continue
        # if self.selection_rect and self.selection_start and self.selection_end:
        #     self.selection_rect = self.ax.axvspan(
        #         self.selection_start, self.selection_end,
        #         facecolor='yellow', alpha=0.3,
        #         edgecolor='orange', linewidth=1
        #     )
        #self.update_selection_rect()

        
                
        # Отрисовка аннотаций
        if len(self.edf_annotations) == 3:  # Проверяем структуру данных
            starts = self.edf_annotations[0]
            durations = self.edf_annotations[1]
            texts = self.edf_annotations[2]
            
            for i in range(len(starts)):
                try:
                    start = float(starts[i])
                    duration = float(durations[i])
                    text = texts[i]
                    end = start + duration if duration > 0 else start
                    
                    # Проверяем видимость аннотации
                    if end < self.time_range[0] or start > self.time_range[1]:
                        continue
                        
                    # Отрисовка в зависимости от типа аннотации
                    if duration > 0:  # Интервальная аннотация
                        rect = self.ax.axvspan(
                            max(start, self.time_range[0]),
                            min(end, self.time_range[1]),
                            facecolor='green', alpha=0.2,
                            edgecolor='darkgreen', linewidth=1
                        )
                        x_pos = (max(start, self.time_range[0]) + min(end, self.time_range[1])) / 2
                    else:  # Точечная аннотация
                        line = self.ax.axvline(
                            x=start,
                            color='green',
                            linestyle='-',
                            alpha=0.7,
                            linewidth=1.5
                        )
                        x_pos = start
                    
                    # Добавляем текст аннотации
                    self.ax.text(
                        x_pos, len(self.signals) - 0.8, text,  # Изменили позицию по Y
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, pad=2)
                    )
                    
                except (ValueError, IndexError) as e:
                    print(f"Ошибка обработки аннотации {i}: {e}")
                    continue
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UltraCompactEEGViewer()
    viewer.show()
    sys.exit(app.exec())