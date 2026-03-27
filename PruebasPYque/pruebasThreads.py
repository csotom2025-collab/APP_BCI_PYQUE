import threading
class PrimerHilo(threading.Thread):
    def __init__(self):
        super().__init__()
        self.n=300
    def run(self):
        for i in range(self.n):
            print("Hilo 1 ", i)
        

class Segundohilo(threading.Thread):
    def __init__(self,):
        self.n=300
        super().__init__()
    def run(self):
        for i in range(self.n):
            print("Hilo Second ", i)



primero = PrimerHilo()
segundo = Segundohilo()

primero.start()
segundo.start()