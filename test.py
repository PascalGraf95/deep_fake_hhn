import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl

class VideoBackgroundWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Background Example")
        self.setGeometry(100, 100, 800, 600)





        # Create a video widget to display the video
        self.video_widget = QVideoWidget()
        self.setCentralWidget(self.video_widget)

        # Create a vertical layout for the central widget
        layout = QVBoxLayout(self.video_widget)

        # Create a media player to control the video
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)

        # Load the video file
        video_path = "./videos/distant_particles_loop.mp4"  # Replace with the path to your video file
        self.media_player.setSource(QUrl.fromLocalFile(video_path))

        # Set the video to loop
        self.media_player.setLoops(-1)  # -1 means infinite loop

        # Play the video
        self.media_player.play()

        # Create a button for demonstration purposes
        button = QPushButton("Click Me!")
        layout.addWidget(button)

def main():
    app = QApplication(sys.argv)
    window = VideoBackgroundWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()