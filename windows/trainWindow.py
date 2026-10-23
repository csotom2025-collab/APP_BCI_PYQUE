import csv
import os
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

MINIMUM_RECORDINGS_PER_CLASS = 30


class TrainingThread(QThread):
    completed = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, training_args, parent=None):
        super().__init__(parent)
        self.training_args = training_args

    def run(self):
        try:
            controllerTraining().train_model(**self.training_args)
        except Exception as error:
            error_traceback = traceback.format_exc()
            print(error_traceback, file=sys.stderr)
            self.failed.emit(str(error))
        else:
            self.completed.emit()


class LDAVisualizationThread(QThread):
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, user, project_root, parent=None):
        super().__init__(parent)
        self.user = user
        self.project_root = Path(project_root)

    def run(self):
        script = (
            self.project_root
            / "models"
            / "newTraining"
            / "visualize_lda_analysis.py"
        )
        output_dir = self.project_root / "trainingOutputs"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(script),
                    "--user",
                    self.user,
                    "--top-cascades",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as error:
            self.failed.emit(f"No se pudo iniciar el visualizador LDA: {error}")
            return

        if result.stdout:
            print(result.stdout)
        if result.returncode:
            details = result.stderr.strip() or result.stdout.strip()
            self.failed.emit(
                details or "El generador de visualizaciones LDA falló."
            )
            return

        figures_dir = (
            output_dir / "figures" / self.user / "top_cascadas"
        )
        self.completed.emit(str(figures_dir))


class DatasetPreparationThread(QThread):
    stage_changed = pyqtSignal(str)
    delete_confirmation_requested = pyqtSignal(int)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, user, project_root, parent=None):
        super().__init__(parent)
        self.user = user
        self.project_root = Path(project_root)
        self.captures_dir = self.project_root / "captures"
        self.quality_dir = (
            self.project_root / "utils" / "chequeo_calidad"
        )
        self._delete_decision = threading.Event()
        self._delete_approved = False

    def set_delete_approved(self, approved):
        self._delete_approved = approved
        self._delete_decision.set()

    def _run_script(self, script_name, *arguments):
        script = self.quality_dir / script_name
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), *map(str, arguments)],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.stdout:
            encoding = sys.stdout.encoding or "utf-8"
            safe_output = result.stdout.encode(
                encoding, errors="replace"
            ).decode(encoding)
            sys.stdout.write(safe_output)
        if result.returncode:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"{script_name} terminó con código {result.returncode}."
                + (f"\n{details}" if details else "")
            )
        return result.stdout

    def _preview_invalid_recordings(self):
        output = self._run_script(
            "remove_invalid_eeg.py",
            self.user,
            "--captures",
            self.captures_dir,
            "--report-dir",
            self.quality_dir,
            "--dry-run",
        )
        match = re.search(r"^Archivos encontrados:\s*(\d+)\s*$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError(
                "No se pudo determinar cuántos EEG inválidos encontró "
                "remove_invalid_eeg."
            )
        return int(match.group(1))

    def _read_balance_report(self):
        report_path = (
            self.quality_dir
            / self.user
            / "reporte_balance_dataset.csv"
        )
        with report_path.open("r", encoding="utf-8-sig", newline="") as report:
            rows = [
                row
                for row in csv.DictReader(report)
                if row.get("usuario") == self.user
            ]
        if not rows:
            raise RuntimeError(
                f"El verificador no generó resultados para {self.user}."
            )
        missing_rows = [
            {
                "clase": row["clase"],
                "faltantes": int(row["faltantes"]),
                "validas": int(row["total_validos"]),
                "minimo": int(row["minimo_requerido"]),
            }
            for row in rows
            if int(row["faltantes"]) > 0
        ]
        return {
            "usuario": self.user,
            "clases_revisadas": len(rows),
            "faltantes": missing_rows,
            "total_faltantes": sum(
                row["faltantes"] for row in missing_rows
            ),
            "reporte": str(report_path),
        }

    def run(self):
        try:
            self.stage_changed.emit("Aplicando filtro de calidad EEG...")
            self._run_script(
                "quality_filter.py",
                self.user,
                "--captures",
                self.captures_dir,
                "--trials-objetivo",
                MINIMUM_RECORDINGS_PER_CLASS,
            )

            self.stage_changed.emit(
                "Revisando grabaciones inválidas..."
            )
            invalid_count = self._preview_invalid_recordings()
            deletion_failures = 0
            if invalid_count:
                self.delete_confirmation_requested.emit(invalid_count)
                self._delete_decision.wait()
                if self._delete_approved:
                    self.stage_changed.emit(
                        "Respaldando y eliminando grabaciones inválidas..."
                    )
                    deletion_output = self._run_script(
                        "remove_invalid_eeg.py",
                        self.user,
                        "--captures",
                        self.captures_dir,
                        "--report-dir",
                        self.quality_dir,
                        "--delete",
                        "--yes",
                        "--backup",
                    )
                    failure_match = re.search(
                        r"^No eliminados por error:\s*(\d+)\s*$",
                        deletion_output,
                        re.MULTILINE,
                    )
                    if failure_match is None:
                        raise RuntimeError(
                            "No se pudo verificar el resultado de la "
                            "eliminación de EEG inválidos."
                        )
                    deletion_failures = int(failure_match.group(1))

            self.stage_changed.emit(
                "Verificando el mínimo de grabaciones por clase..."
            )
            self._run_script(
                "dataset_balance_checker.py",
                self.user,
                "--captures",
                self.captures_dir,
                "--output-dir",
                self.quality_dir,
                "--minimum",
                MINIMUM_RECORDINGS_PER_CLASS,
            )
            result = self._read_balance_report()
            result["eliminacion_aprobada"] = (
                invalid_count == 0 or self._delete_approved
            )
            result["fallidos_eliminacion"] = deletion_failures
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


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

        self.training_thread = None
        self.dataset_preparation_thread = None
        self.lda_visualization_thread = None
        self.setup_ui()
        self.models= ['lstm', 'lda', 'svm', 'random_forest', 'xgboost']
        self.show_models()
        self.show_users()
        
    def setup_ui(self):
        self.setWindowTitle("Ventana de Entrenamiento")
        self.resize(1000, 590)
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Ventana de Entrenamiento"))

        content_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        self.user_combobox = QComboBox()
        self.combo_box_models = QComboBox()
        self.button_start_training = QPushButton("Iniciar entrenamiento")
        self.button_start_training.clicked.connect(self.start_training)
        self.button_start_training.setMinimumHeight(48)
        self.button_check_dataset = QPushButton(
            "Verificar grabaciones para entrenar"
        )
        self.button_check_dataset.clicked.connect(self.check_dataset)
        self.button_check_dataset.setMinimumHeight(48)
        self.button_visualize_lda = QPushButton(
            "Generar visualizaciones LDA (Top cascadas)"
        )
        self.button_visualize_lda.clicked.connect(
            self.start_lda_visualization
        )
        self.button_visualize_lda.setMinimumHeight(48)

        training_group = QGroupBox("Entrenamiento")
        training_layout = QGridLayout(training_group)
        training_layout.addWidget(QLabel("Seleccione usuario:"), 0, 0)
        training_layout.addWidget(self.user_combobox, 0, 1)
        training_layout.addWidget(QLabel("Selecciona un modelo:"), 1, 0)
        training_layout.addWidget(self.combo_box_models, 1, 1)
        training_layout.addWidget(self.button_start_training, 2, 0, 1, 2)
        training_layout.addWidget(self.button_check_dataset, 3, 0, 1, 2)
        training_layout.addWidget(self.button_visualize_lda, 4, 0, 1, 2)
        left_layout.addWidget(training_group)

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
        left_layout.addWidget(signal_group)

        command_group = QGroupBox("Tipos de Comando")
        command_layout = QGridLayout(command_group)
        self.command_checkboxes = {}
        command_types = ["Letters", "Numbers", "Controls"]
        for index, command_type in enumerate(command_types):
            checkbox = QCheckBox(command_type)
            checkbox.setChecked(True)
            self.command_checkboxes[command_type] = checkbox
            command_layout.addWidget(checkbox, 0, index % 3)
        left_layout.addWidget(command_group)
        left_layout.addStretch()
        content_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        channels_group = QGroupBox("Canales EEG")
        channels_layout = QVBoxLayout(channels_group)
        self.channel_map = ChannelMapWidget(training_config.CHANNEL_NAMES)
        self.channel_checkboxes = self.channel_map.checkboxes
        channels_layout.addWidget(
            self.channel_map, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        channels_layout.addWidget(QLabel("Azul: incluido    Gris: excluido"))
        manual_channels_label = QLabel(
            "O escribe los nombres de los canales separados por comas:"
        )
        manual_channels_label.setWordWrap(True)
        manual_channels_label.setMaximumWidth(400)
        channels_layout.addWidget(manual_channels_label)
        self.channels_input = QLineEdit()
        self.channels_input.setPlaceholderText("Ejemplo: F3, FC5, Pz, Oz")
        self.channels_input.setToolTip(
            "Si escribes canales aquí, reemplazarán la selección del diagrama. "
            "Los nombres deben coincidir con las columnas de los archivos CSV."
        )
        channels_layout.addWidget(self.channels_input)
        right_layout.addWidget(channels_group)
        right_layout.addStretch()
        content_layout.addLayout(right_layout, 1)

        main_layout.addLayout(content_layout)
        self.training_status_label = QLabel("")
        main_layout.addWidget(self.training_status_label)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
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
        manual_channels = self.channels_input.text().strip()
        if manual_channels:
            channels = [channel.strip() for channel in manual_channels.split(",")]
            if any(not channel for channel in channels):
                return []
            return list(dict.fromkeys(channels))

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
        if (
            self.training_thread is not None
            or self.dataset_preparation_thread is not None
            or self.lda_visualization_thread is not None
        ):
            return

        user = self.get_user()
        if not user:
            QMessageBox.warning(self, "Usuario requerido", "Selecciona un usuario.")
            return

        path = "captures"
        model = self.get_model()
        channels = self.getSelectedChannels()
        command_types = self.getCommandTypes()

        if not channels:
            QMessageBox.warning(
                self,
                "Canales requeridos",
                "Selecciona al menos un canal o revisa la lista separada por comas.",
            )
            return

        print(f"Entrenando modelo {model} con los datos de {user} que se encuentran en la ruta: {path}")
        if not command_types:
            QMessageBox.warning(
                self,
                "Comandos requeridos",
                "Selecciona al menos un tipo de comando para entrenar.",
            )
            return

        training_args = {
            "user": user,
            "dataPath": path,
            "modelType": model,
            "output_dir": "trainingOutputs",
            "commandTypes": command_types,
            "channels": channels,
            "fs": self.getSelectedFs(),
            "use_p300window": self.getUseP300Window(),
            "apply_baseline_correction": self.getApplyBaselineCorrection(),
            "seed": self.getSelectedSeed(),
        }
        self.button_start_training.setEnabled(False)
        self.button_check_dataset.setEnabled(False)
        self.button_visualize_lda.setEnabled(False)
        self.training_status_label.setText("Entrenamiento en curso...")
        self.training_thread = TrainingThread(training_args, self)
        self.training_thread.completed.connect(self._on_training_completed)
        self.training_thread.failed.connect(self._on_training_failed)
        self.training_thread.finished.connect(self._on_training_thread_finished)
        self.training_thread.start()

    def check_dataset(self):
        if (
            self.training_thread is not None
            or self.dataset_preparation_thread is not None
            or self.lda_visualization_thread is not None
        ):
            return

        user = self.get_user()
        if not user:
            QMessageBox.warning(self, "Usuario requerido", "Selecciona un usuario.")
            return

        project_root = Path("captures").resolve().parent
        self.button_start_training.setEnabled(False)
        self.button_check_dataset.setEnabled(False)
        self.button_visualize_lda.setEnabled(False)
        self.user_combobox.setEnabled(False)
        self.training_status_label.setText(
            "Preparando y verificando las grabaciones..."
        )
        self.dataset_preparation_thread = DatasetPreparationThread(
            user, project_root, self
        )
        self.dataset_preparation_thread.stage_changed.connect(
            self.training_status_label.setText
        )
        self.dataset_preparation_thread.delete_confirmation_requested.connect(
            self._confirm_invalid_recording_removal
        )
        self.dataset_preparation_thread.completed.connect(
            self._on_dataset_preparation_completed
        )
        self.dataset_preparation_thread.failed.connect(
            self._on_dataset_preparation_failed
        )
        self.dataset_preparation_thread.finished.connect(
            self._on_dataset_preparation_finished
        )
        self.dataset_preparation_thread.start()

    def start_lda_visualization(self):
        if (
            self.training_thread is not None
            or self.dataset_preparation_thread is not None
            or self.lda_visualization_thread is not None
        ):
            return

        user = self.get_user()
        if not user:
            QMessageBox.warning(
                self, "Usuario requerido", "Selecciona un usuario."
            )
            return

        project_root = Path(__file__).resolve().parents[1]
        self.button_start_training.setEnabled(False)
        self.button_check_dataset.setEnabled(False)
        self.button_visualize_lda.setEnabled(False)
        self.user_combobox.setEnabled(False)
        self.training_status_label.setText(
            f"Generando visualizaciones LDA con las top cascadas de {user}..."
        )
        self.lda_visualization_thread = LDAVisualizationThread(
            user, project_root, self
        )
        self.lda_visualization_thread.completed.connect(
            self._on_lda_visualization_completed
        )
        self.lda_visualization_thread.failed.connect(
            self._on_lda_visualization_failed
        )
        self.lda_visualization_thread.finished.connect(
            self._on_lda_visualization_finished
        )
        self.lda_visualization_thread.start()

    def _on_lda_visualization_completed(self, figures_dir):
        self.training_status_label.setText(
            "Visualizaciones LDA de top cascadas completadas."
        )
        QMessageBox.information(
            self,
            "Visualizaciones completadas",
            f"Se generaron las visualizaciones LDA para {self.get_user()} "
            f"con los cinco modelos top.\n\nDirectorio:\n{figures_dir}",
        )

    def _on_lda_visualization_failed(self, error_message):
        self.training_status_label.setText(
            "La generación de visualizaciones LDA terminó con errores."
        )
        QMessageBox.critical(
            self,
            "Error al generar visualizaciones LDA",
            f"{error_message}\n\nConsulta la salida de la aplicación para más detalles.",
        )

    def _on_lda_visualization_finished(self):
        if self.lda_visualization_thread is not None:
            self.lda_visualization_thread.deleteLater()
            self.lda_visualization_thread = None
        self.user_combobox.setEnabled(True)
        self.button_start_training.setEnabled(True)
        self.button_check_dataset.setEnabled(True)
        self.button_visualize_lda.setEnabled(True)

    def _confirm_invalid_recording_removal(self, invalid_count):
        user = self.get_user()
        backup_dir = (
            Path("captures").resolve().parent
            / "backup_eliminados"
            / user
        )
        answer = QMessageBox.question(
            self,
            "Confirmar eliminación de EEG inválidos",
            f"Se encontraron {invalid_count} grabaciones clasificadas como "
            f"inválidas para {user}.\n\n"
            "¿Deseas eliminarlas? Se guardará un respaldo antes de cada "
            f"eliminación en:\n{backup_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self.dataset_preparation_thread.set_delete_approved(
            answer == QMessageBox.StandardButton.Yes
        )

    def _on_dataset_preparation_completed(self, result):
        user = result["usuario"]
        minimum = MINIMUM_RECORDINGS_PER_CLASS
        total_missing = result["total_faltantes"]
        if total_missing:
            details = "\n".join(
                f"• {row['clase']}: faltan {row['faltantes']} "
                f"({row['validas']}/{row['minimo']})"
                for row in result["faltantes"]
            )
            title = "Faltan grabaciones"
            message = (
                f"{user} todavía no cumple el mínimo de {minimum} "
                "grabaciones válidas por clase:\n\n"
                f"{details}\n\n"
                f"Total faltante: {total_missing}"
            )
        else:
            title = "Dataset listo para entrenar"
            message = (
                f"{user} cumple el mínimo de {minimum} grabaciones válidas "
                f"en las {result['clases_revisadas']} clases revisadas."
            )

        if not result["eliminacion_aprobada"]:
            message += (
                "\n\nNo se eliminaron los EEG inválidos; se conservaron "
                "los archivos originales."
            )
        if result["fallidos_eliminacion"]:
            message += (
                "\n\nAdvertencia: no se pudieron eliminar "
                f"{result['fallidos_eliminacion']} archivos inválidos. "
                "Consulta los reportes de remove_invalid_eeg."
            )
        message += f"\n\nReporte: {result['reporte']}"
        self.training_status_label.setText(title)
        if total_missing:
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def _on_dataset_preparation_failed(self, error_message):
        self.training_status_label.setText(
            "La verificación de grabaciones terminó con errores."
        )
        QMessageBox.critical(
            self,
            "Error al verificar el dataset",
            f"{error_message}\n\nConsulta la salida de la aplicación para más detalles.",
        )

    def _on_dataset_preparation_finished(self):
        if self.dataset_preparation_thread is not None:
            self.dataset_preparation_thread.deleteLater()
            self.dataset_preparation_thread = None
        self.user_combobox.setEnabled(True)
        self.button_start_training.setEnabled(True)
        self.button_check_dataset.setEnabled(True)
        self.button_visualize_lda.setEnabled(True)

    def _on_training_completed(self):
        self.training_status_label.setText("Entrenamiento completado.")

    def _on_training_failed(self, error_message):
        self.training_status_label.setText("El entrenamiento terminó con errores.")
        QMessageBox.critical(
            self,
            "Error de entrenamiento",
            f"{error_message}\n\nConsulta la salida de la aplicación para más detalles.",
        )

    def _on_training_thread_finished(self):
        self.button_start_training.setEnabled(True)
        self.button_visualize_lda.setEnabled(True)
        if self.training_thread is not None:
            self.training_thread.deleteLater()
            self.training_thread = None
        if self.dataset_preparation_thread is None:
            self.button_check_dataset.setEnabled(True)
        self.button_start_training.setEnabled(
            self.dataset_preparation_thread is None
        )

    def closeEvent(self, event):
        running_threads = (
            self.training_thread,
            self.dataset_preparation_thread,
            self.lda_visualization_thread,
        )
        if any(thread is not None and thread.isRunning() for thread in running_threads):
            QMessageBox.information(
                self,
                "Proceso en curso",
                "Espera a que termine el proceso antes de cerrar esta ventana.",
            )
            event.ignore()
            return
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainWindow()
    window.show()
    sys.exit(app.exec())