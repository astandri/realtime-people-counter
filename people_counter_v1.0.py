import argparse
import os
import time
import uuid
from datetime import datetime

import cv2
import dlib
import imagezmq
import imutils
import numpy as np
from flask import Flask, Response, render_template
from imutils.video import FPS, VideoStream

from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from utils.utils import read_config

app = Flask(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
MIN_CONFIDECE = 0.45
SKIP_FRAMES = 2
MAX_DISAPPEARED = 10
MAX_DISTANCE = 100
FACE_MODEL = "models/face_detector.caffemodel"
FACE_PROTO = "models/face_detector.prototxt"
PERSON_MODEL = "models/person_detector.caffemodel"
PERSON_PROTO = "models/person_detector.prototxt"
ESTIMATED_NUM_PIS = 1
ACTIVE_CHECK_PERIOD = 1
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD


def generate():

    # load our serialized model from disk
    print("[INFO] loading model...")
    # net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    net = cv2.dnn.readNetFromCaffe(PERSON_PROTO, PERSON_MODEL)

    configs = read_config("configs.json")
    imageHub = imagezmq.ImageHub()
    # print("[INFO] opening video Stream from {}".format(configs["urls"][0]))
    # vs = cv2.VideoCapture(configs["urls"][0], cv2.CAP_FFMPEG)

    lastActive = {}
    lastActiveCheck = datetime.now()

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=MAX_DISAPPEARED, maxDistance=MAX_DISTANCE)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    person_in = 0
    person_out = 0

    # start the frames per second throughput estimator
    fps = 0

    # loop over frames from the video stream
    while True:

        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        configs = read_config("configs.json")

        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b"OK")
        # if a device is not in the last active dictionary then it means
        # that its a newly connected device
        if rpiName not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))
        # record the last active time for the device from which we just
        # received a frame
        lastActive[rpiName] = datetime.now()

        frame, original_frame = (
            imutils.resize(frame, width=1024),
            imutils.resize(frame, width=1024),
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            "appsrc  ! h264parse ! "
            "rtph264pay config-interval=1 pt=96 ! "
            "gdppay ! tcpserversink host=localhost port=5000 ",
            fourcc,
            20.0,
            (640, 480),
        )

        rects = []

        t1 = time.time()

        # Detection Area
        startX_area = int(configs["startX_area"] * W)
        endX_area = int(configs["endX_area"] * W)
        startY_area = int(configs["startY_area"] * H)
        endY_area = int(configs["endY_area"] * H)

        cv2.rectangle(
            frame, (startX_area, startY_area), (endX_area, endY_area), (0, 0, 255), 2
        )

        if totalFrames % configs["skip_frames"] == 0:

            trackers = []

            # blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0)) # for face
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)  # for person
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > configs["min_confidence"]:

                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # reduce rectange size
                    x_reduce_size = 0.4 * (endX - startX) / 2
                    y_reduce_size = 0.75 * (endY - startY)
                    startX = int(startX + (x_reduce_size * 0.75))
                    endX = int(endX - (x_reduce_size * 1.25))
                    endY = int(endY - y_reduce_size)

                    # face = frame[startY:endY, startX:endX]
                    # # ensure the face ROI is sufficiently large
                    # if face.shape[0] < 20 or face.shape[1] < 20:
                    # 	continue

                    if (startX >= startX_area and startY >= startY_area) and (
                        endX <= endX_area and endY <= endY_area
                    ):

                        # only track if face detected inside a box
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(
                            int(startX), int(startY), int(endX), int(endY)
                        )
                        tracker.start_track(rgb, rect)

                        trackers.append(tracker)

                        cv2.rectangle(
                            frame, (startX, startY), (endX, endY), (0, 0, 255), 2
                        )

        else:
            # loop over the trackers
            for tracker in trackers:

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid), rect in zip(objects.items(), rects):
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
                # cv2.imwrite(
                # 	'faces/{}_{:.2f}.jpg'.format(uuid.uuid4(), confidence),
                # 	original_frame[rect[1]:rect[3], rect[0]:rect[2]]
                # )

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                x = [c[0] for c in to.centroids]
                y = [c[1] for c in to.centroids]
                direction_x = centroid[0] - np.mean(x)
                direction_y = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted_in:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction_y > 0:
                        person_in += 1
                        to.counted_in = True

                if not to.counted_out:
                    if direction_y < 0:
                        person_out += 1
                        to.counted_out = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(
                frame,
                text,
                (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
        # 					cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # construct a tuple of information we will be displaying on the
        # frame
        info = [("Jumlah Orang Masuk", person_in), ("Jumlah Orang Keluar", person_out)]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(
                frame,
                text,
                (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # # check to see if we should write the frame to disk
        # if writer is not None:
        #     writer.write(frame)

        # # show the output frame
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1

        # if current time *minus* last time when the active device check
        # was made is greater than the threshold set then do a check
        if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
            # loop over all previously active devices
            for (rpiName, ts) in list(lastActive.items()):
                # remove the RPi from the last active and frame
                # dictionaries if the device hasn't been active recently
                if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                    print("[INFO] lost connection to {}".format(rpiName))
                    lastActive.pop(rpiName)
            # set the last active check time as current time
            lastActiveCheck = datetime.now()
        # if the `q` key was pressed, break from the loop

        encodedFrame = cv2.imencode(".jpg", frame)[1].tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedFrame) + b"\r\n"
        )

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == "__main__":
    # start the flask app
    app.run(host="localhost", port=5000, debug=False, threaded=True, use_reloader=False)
