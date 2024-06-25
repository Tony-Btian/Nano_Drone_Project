from PySide6.QtWidgets import QTextEdit

class MessageDisplay:
    def __init__(self, text_edit: QTextEdit):
        self.text_edit = text_edit

    def set_text(self, text: str):
        self.text_edit.setPlainText(text)

    def append_text(self, text: str):
        self.text_edit.append(text)

    def loading_animation(self, step: int):
        animation_steps = ['.', '..', '...', '....', '.....']
        loading_text = f"Loading MiDaS model, please wait{animation_steps[step % len(animation_steps)]}"
        self.set_text(loading_text)