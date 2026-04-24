import numpy as np
class TextFieldController:
    def __init__(self, text_field):
        self.text_field = text_field

    def add_character(self, char):
        self.text_field.add_character(char)

    def add_new_line(self):
        self.text_field.add_new_line()

    def remove_last_character(self):
        self.text_field.remove_last_character()

    def space(self):
        self.text_field.space()

    def clear(self):
        self.text_field.set_text("")
    def predict_next_character(self):
        GRID = ["A", "E", "I", "O", "U","1","2","3",
                "S", "R", "N", "L", "D","4","5","6",
                "C", "T", "M", "P", "B","7","8","9",
                "G", "V", "Y", "Q", "H","0","───", "⟵",
                "F", "Z", "J", "Ñ", "X","K ", "W","↩"]          
        next_char = GRID[np.random.randint(0, len(GRID))]
        self.next_character(next_char)  
        # Aquí podrías implementar un modelo de predicción basado en el texto actual
        # Por ejemplo, podrías usar un modelo de lenguaje para predecir el siguiente carácter
        pass
    def next_character(self, char):
        # Aquí podrías implementar la lógica para mostrar el siguiente carácter sugerido
        print(f"El siguiente carácter sugerido es: {char}")
        if char  == "───":
            self.space()
        elif char  == "⟵":
            self.remove_last_character()
        elif char  == "↩":
            self.add_new_line()
        else:
            self.add_character(char)
    