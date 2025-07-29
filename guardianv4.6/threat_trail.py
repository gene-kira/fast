from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QTimer, QPointF, QPropertyAnimation
from PyQt5.QtGui import QBrush, QColor, QPen, QFont
import random

class ThreatTrailCanvas(QGraphicsView):
    def __init__(self, width=600, height=400):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")
        self.setSceneRect(0, 0, width, height)

        self.ip_trails = {}
        self._draw_grid()

    def _draw_grid(self):
        grid_size = 40
        pen = QPen(QColor("#222222"), 1)
        for x in range(0, int(self.sceneRect().width()), grid_size):
            self.scene().addLine(x, 0, x, self.sceneRect().height(), pen)
        for y in range(0, int(self.sceneRect().height()), grid_size):
            self.scene().addLine(0, y, self.sceneRect().width(), y, pen)

    def add_ip_trace(self, ip, points, avatar_name="Unknown"):
        trail_color = self.get_avatar_color(avatar_name)
        for i, point in enumerate(points):
            glow = max(255 - i * 30, 60)
            color = QColor(trail_color.red(), trail_color.green(), trail_color.blue(), glow)
            dot = QGraphicsEllipseItem(0, 0, 10, 10)
            dot.setBrush(QBrush(color))
            dot.setPen(QPen(Qt.NoPen))
            dot.setPos(point)
            self.scene().addItem(dot)
        self.spawn_ripple(points[-1], trail_color)
        self.start_countdown(points[-1])
        self.log_lore(ip, avatar_name, points[-1])

    def spawn_ripple(self, center, color):
        ripple = QGraphicsEllipseItem(-20, -20, 40, 40)
        ripple.setBrush(Qt.NoBrush)
        ripple.setPen(QPen(color, 2))
        ripple.setPos(center)
        self.scene().addItem(ripple)

        ripple_anim = QPropertyAnimation(ripple, b"scale")
        ripple_anim.setDuration(800)
        ripple_anim.setStartValue(1)
        ripple_anim.setEndValue(2)
        ripple_anim.start()
        QTimer.singleShot(900, lambda: self.scene().removeItem(ripple))

    def start_countdown(self, pos, seconds=3):
        text = QGraphicsTextItem(f"Breach: {seconds}")
        text.setFont(QFont("Consolas", 10))
        text.setDefaultTextColor(QColor("#ff6600"))
        text.setPos(pos + QPointF(20, -10))
        self.scene().addItem(text)

        def update():
            nonlocal seconds
            seconds -= 1
            if seconds >= 0:
                text.setPlainText(f"Breach: {seconds}")
            else:
                self.scene().removeItem(text)
                timer.stop()

        timer = QTimer()
        timer.timeout.connect(update)
        timer.start(1000)

    def log_lore(self, ip, avatar, pos):
        x = int(pos.x())
        y = int(pos.y())
        print(f"[LORE] {avatar} initiated breach path at ({x},{y}) from {ip}")

    def get_avatar_color(self, avatar):
        themes = {
            "Phantom Crawler": QColor("#ff3300"),
            "Signal Wolf": QColor("#ffffff"),
            "Data Wraith": QColor("#00ffee"),
            "Hive Spectre": QColor("#cc00ff"),
            "Null Echo": QColor("#00cc66"),
        }
        return themes.get(avatar, QColor("#ff9900"))

