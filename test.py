import cv2

from utils.utils import read_config

configs = read_config("configs.json")

video = cv2.VideoCapture("rtsp://localhost:8554/")

while True:
    _, frame = video.read()
    cv2.imshow("RTSP", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
