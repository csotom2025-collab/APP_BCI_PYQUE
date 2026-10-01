import os
import sys
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from controllers.TrainingController import controllerTraining
from models.newTraining import config as training_config


CHANNEL_POSITIONS = {
    "AF3": (120, 111),
    "F7": (77, 126),
    "FC5": (79, 171),
    "F3": (120, 171),
    "T7": (51, 200),
    "P7": (87, 284),
    "O1": (120, 330),
    "O2": (190, 330),
    "P8": (222, 284),
    "T8": (259, 200),
    "F8": (231, 126),
    "FC6": (230, 171),
    "F4": (188, 171),
    "AF4": (189, 111),
}

CHANNEL_CONNECTIONS = (
    ("AF3", "F7"), ("AF3", "F3"), ("AF3", "AF4"), ("F7", "FC5"),
    ("F7", "T7"), ("FC5", "F3"), ("FC5", "T7"), ("F3", "F4"),
    ("F4", "FC6"), ("F4", "AF4"), ("AF4", "F8"), ("FC6", "T8"),
    ("F8", "T8"), ("T7", "P7"), ("P7", "O1"), ("O1", "O2"),
    ("O2", "P8"), ("P8", "T8"),
)


class ChannelNodeCheckBox(QCheckBox):
    def __init__(self, channel):
        super().__init__(channel)
        self.setFixedSize(38, 38)
        self.setAccessibleName(f"Canal {channel}")
        self.setToolTip(f"Seleccionar o quitar el canal {channel}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = self.isChecked()
        fill_color = QColor("#d7edfa" if selected else "#d4d8de")
        border_color = QColor("#1675a8" if selected else "#777f89")
        painter.setPen(QPen(border_color, 3 if selected else 2))
        painter.setBrush(fill_color)
        painter.drawEllipse(self.rect().adjusted(3, 3, -3, -3))
        event.accept()


class ChannelMapWidget(QWidget):
    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setMinimumSize(318, 376)
        self.checkboxes = {}
        self.channel_labels = {}
        for channel in channel_names:
            checkbox = ChannelNodeCheckBox(channel)
            checkbox.setChecked(True)
            self.checkboxes[channel] = checkbox
            checkbox.setParent(self)
            label = QLabel(channel, self)
            label.setFixedSize(32, 32)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.setStyleSheet(
                "background: transparent; color: #111827;"
            )
            label.raise_()
            self.channel_labels[channel] = label
        self._position_checkboxes()

    def sizeHint(self):
        return QSize(318, 376)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_checkboxes()

    def _position_checkboxes(self):
        scale = min(self.width() / 318, self.height() / 376)
        offset_x = (self.width() - 318 * scale) / 2
        offset_y = (self.height() - 376 * scale) / 2
        for channel, checkbox in self.checkboxes.items():
            x, y = CHANNEL_POSITIONS[channel]
            checkbox.move(
                round(offset_x + x * scale - checkbox.width() / 2),
                round(offset_y + y * scale - checkbox.height() / 2),
            )
            label = self.channel_labels[channel]
            label.move(
                round(offset_x + x * scale - label.width() / 2),
                round(offset_y + y * scale - label.height() / 2),
            )
            label.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7faff"))

        scale = min(self.width() / 318, self.height() / 376)
        offset_x = (self.width() - 318 * scale) / 2
        offset_y = (self.height() - 376 * scale) / 2
        painter.translate(offset_x, offset_y)
        painter.scale(scale, scale)

        gradient = QLinearGradient(50, 42, 260, 345)
        gradient.setColorAt(0, QColor("#c8def1"))
        gradient.setColorAt(1, QColor("#edf4fa"))
        head = QPainterPath()
        head.moveTo(145, 27)
        head.lineTo(158, 47)
        head.cubicTo(218, 54, 275, 105, 276, 190)
        head.cubicTo(279, 271, 229, 339, 159, 351)
        head.cubicTo(88, 343, 39, 277, 42, 190)
        head.cubicTo(45, 105, 97, 55, 145, 47)
        head.closeSubpath()
        painter.setPen(QPen(QColor("#58758e"), 2))
        painter.setBrush(gradient)
        painter.drawPath(head)

        painter.setBrush(QColor("#d7e5f1"))
        painter.drawEllipse(31, 178, 20, 46)
        painter.drawEllipse(267, 178, 20, 46)

        painter.setPen(QPen(QColor("#536b80"), 1.5))
        for first, second in CHANNEL_CONNECTIONS:
            x1, y1 = CHANNEL_POSITIONS[first]
            x2, y2 = CHANNEL_POSITIONS[second]
            painter.drawLine(x1, y1, x2, y2)
        event.accept()



class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setup_ui()
        # self.data = {'user':'pathUser', 'MVP':'pathMVP', 'anodaUser':'pathAnodaUser'}
        # self.show_users()
        self.models= ['lstm', 'lda', 'svm', 'random_forest', 'xgboost']
        self.show_models()
        self.show_users()
        
    def setup_ui(self):
        self.setWindowTitle("Ventana de Entrenamiento")
        layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.user_combobox = QComboBox()
        self.combo_box_models = QComboBox()
        self.button_start_training = QPushButton("Iniciar entrenamiento")
        self.button_start_training.clicked.connect(self.start_training)

        layout.addWidget(QLabel("Ventana de Entrenamiento"))
        self.grid_layout.addWidget(QLabel("Seleccione usuario:"), 0, 0)
        self.grid_layout.addWidget(self.user_combobox, 0, 1)
        self.grid_layout.addWidget(QLabel("Selecciona un modelo:"), 1, 0)
        self.grid_layout.addWidget(self.combo_box_models, 1, 1)

        layout.addLayout(self.grid_layout)

        signal_group = QGroupBox("Configuración de señal")
        signal_layout = QGridLayout(signal_group)

        self.use_p300_window_checkbox = QCheckBox("Usar ventana P300")
        self.use_p300_window_checkbox.setChecked(
            training_config.USE_P300_WINDOW_ONLY
        )
        self.apply_baseline_correction_checkbox = QCheckBox(
            "Aplicar corrección de línea base"
        )
        self.apply_baseline_correction_checkbox.setChecked(
            training_config.APPLY_BASELINE_CORRECTION
        )
        self.fs_spinbox = QSpinBox()
        self.fs_spinbox.setRange(1, 4096)
        self.fs_spinbox.setValue(128)

        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 9999)
        self.seed_spinbox.setValue(42)

        signal_layout.addWidget(self.use_p300_window_checkbox, 0, 0, 1, 2)
        signal_layout.addWidget(
            self.apply_baseline_correction_checkbox, 1, 0, 1, 2
        )
        signal_layout.addWidget(QLabel("Frecuencia de muestreo (Hz):"), 2, 0)
        signal_layout.addWidget(self.fs_spinbox, 2, 1)
        signal_layout.addWidget(QLabel("Semilla (Seed):"), 3, 0)
        signal_layout.addWidget(self.seed_spinbox, 3, 1)
        layout.addWidget(signal_group)

        channels_group = QGroupBox("Canales EEG (Emotiv)")
        channels_layout = QVBoxLayout(channels_group)
        self.channel_map = ChannelMapWidget(training_config.CHANNEL_NAMES)
        self.channel_checkboxes = self.channel_map.checkboxes
        channels_layout.addWidget(
            self.channel_map, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        channels_layout.addWidget(QLabel("Azul: incluido    Gris: excluido"))

        command_group = QGroupBox("Tipos de Comando")
        command_layout = QGridLayout(command_group)
        self.command_checkboxes = {}
        command_types = ["Letters", "Numbers", "Controls"]
        for index, command_type in enumerate(command_types):
            checkbox = QCheckBox(command_type)
            checkbox.setChecked(True)
            self.command_checkboxes[command_type] = checkbox
            command_layout.addWidget(checkbox, 0, index % 3)
        layout.addWidget(channels_group)
        layout.addWidget(command_group)
        layout.addWidget(self.button_start_training)

        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)
        self.move(200, 200)

    

    

    def show_models(self):
        self.combo_box_models.clear()
        self.combo_box_models.addItems(self.models)

    def get_model(self):
        model = self.combo_box_models.currentText()
        return model
    
    def get_user(self):
        user_id = self.user_combobox.currentText()
        return user_id
    
    def get_path(self):
        user = self.get_user()
        if not user:
            return ""
        return os.path.join("captures", f"{user}")
    def show_users(self):
        self.user_combobox.clear()
        captures_dir = "captures"
        if not os.path.exists(captures_dir):
            QMessageBox.warning(self, "Directorio no encontrado", f"No se encontró el directorio '{captures_dir}'.")
            return

        users = [d for d in os.listdir(captures_dir) if os.path.isdir(os.path.join(captures_dir, d)) and d.startswith("User")]
        if not users:
            QMessageBox.information(self, "Usuarios no encontrados", "No se encontraron carpetas de usuarios en el directorio 'captures'.")
            return

        self.user_combobox.addItems(users)
    def getSelectedChannels(self):
        return [
            channel
            for channel, checkbox in self.channel_checkboxes.items()
            if checkbox.isChecked()
        ]

    def getSelectedFs(self):
        return self.fs_spinbox.value()

    def getUseP300Window(self):
        return self.use_p300_window_checkbox.isChecked()

    def getApplyBaselineCorrection(self):
        return self.apply_baseline_correction_checkbox.isChecked()
    def getCommandTypes(self):
        return [
            command_type
            for command_type, checkbox in self.command_checkboxes.items()
            if checkbox.isChecked()
        ]
    def getSelectedSeed(self):
        return self.seed_spinbox.value()
    def start_training(self):
        user = self.get_user()
        path = "captures" 
        model = self.get_model()

        if not self.getSelectedChannels():
            QMessageBox.warning(
                self,
                "Canales requeridos",
                "Selecciona al menos un canal EEG para entrenar el modelo.",
            )
            return

        print(f"Entrenando modelo {model} con los datos de {user} que se encuentran en la ruta: {path}")
        training_controller = controllerTraining()
        training_controller.train_model(
            user,
            path,
            model,
            output_dir="trainingOutputs",
            commandTypes=self.getCommandTypes(),
            channels=self.getSelectedChannels(),
            fs=self.getSelectedFs(),
            use_p300window=self.getUseP300Window(),
            apply_baseline_correction=self.getApplyBaselineCorrection(),
            seed=self.getSelectedSeed()
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainWindow()
    window.show()
    sys.exit(app.exec())