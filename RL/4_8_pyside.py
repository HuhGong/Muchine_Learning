# 4_8_pyside.py
# pyqt   (유료)
# pyside (무료)

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        self.label = QLabel()

        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)

        layout = QVBoxLayout()
        # layout = QHBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        self.inner = QHBoxLayout()
        self.name = QLabel('my name')
        self.age = QLabel('my age')
        self.inner.addWidget(self.name)
        self.inner.addWidget(self.age)

        layout.addLayout(self.inner)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
