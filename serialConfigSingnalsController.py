
class ControllerSerialConfig:
    def __init__(self, menu_window):
        self.menu_window = menu_window
    def update_serial_config(self, port, baudrate):
        self.menu_window.update_serial_config(port, baudrate)
