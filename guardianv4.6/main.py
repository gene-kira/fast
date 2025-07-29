import autoloader
autoloader.autoload()

from PyQt5.QtWidgets import QApplication
from guardian_gui import GuardianGUI
import sys

def main():
    app = QApplication(sys.argv)
    gui = GuardianGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

