import sys

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Pedalboard Simulation")

        button = QPushButton("Press Me!")

        self.setFixedSize(QSize(680, 480))

        self.setCentralWidget(button)


print("Starting Core Application")
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
print("Closing Application. GoodBye!!")
