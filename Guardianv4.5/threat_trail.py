from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QTimer, QPointF, QPropertyAnimation
from PyQt5.QtGui import QBrush, QColor, QPen, QFont

class ThreatTrailCanvas(QGraphicsView):
    def __init__(self, width=600, height=400):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")
        self.setSceneRect(0, 0, width, height)

        self.ip_trails = {}
        self.trail_colors = {}
        self._draw_grid()

    def _draw_grid(self):
        grid_size = 40
        pen = QPen(QColor("#222222"), 1)
        for x in range(0, int(self.sceneRect().width()), grid_size):
            self.scene().addLine(x, 0, x, self.sceneRect().height(), pen)
        for y in range(0, int(self.sceneRect().height()), grid_size):
            self.scene().addLine(0, y, self.sceneRect().width(), y, pen)

    def add_ip_trace(self, ip, path_points):
        color = QColor("#ff6600")
        self.trail_colors[ip] = color

        for i, point in enumerate(path_points):
            glow_intensity = 255 - (i * 15)
            dot_color = QColor(color.red(), color.green(), color.blue(), max(glow_intensity, 60))
            dot = QGraphicsEllipseItem(0, 0, 10, 10)
            dot.setBrush(QBrush(dot_color))
            dot.setPen(QPen(Qt.NoPen))
            dot.setPos(point)
            self.scene().addItem(dot)

class RippleBurst:
    def __init__(self, scene, center):
        self.scene = scene
        self.center = center
        self.anim_item = QGraphicsEllipseItem(-25, -25, 50, 50)
        self.anim_item.setBrush(QBrush(Qt.NoBrush))
        self.anim_item.setPen(QPen(QColor("#ff3300"), 2))
        self.anim_item.setPos(center)
        self.scene.addItem(self.anim_item)

        self.animation = QPropertyAnimation(self.anim_item, b"scale")
        self.animation.setDuration(1200)
        self.animation.setStartValue(1)
        self.animation.setEndValue(3)
        self.animation.start()

        QTimer.singleShot(1400, lambda: self.scene.removeItem(self.anim_item))

class BreachCountdown:
    def __init__(self, scene, ip_pos, seconds):
        self.scene = scene
        self.counter = seconds
        self.text_item = QGraphicsTextItem(f"Breach: {self.counter}")
        self.text_item.setFont(QFont("OCR A Extended", 10

