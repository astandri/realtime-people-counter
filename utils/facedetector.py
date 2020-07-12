from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

class FaceDetector:

    def __init__(self, proto, model, input_path=None, output_path=None, confidence=0.75, skipframes=30):

        self.face_detector_model = cv2.dnn.readNetFromCaffe(proto, model)
        self.input_path = input_path
        self.output_path = output_path
        self.confidence = 0.6
        self.skipframes = skipframes
        
        self.writer = None
        self.W = None
        self.H = None

        self.ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        self.totalFrames = 0
        self.totalvisitor = 0

        self.fps = FPS().start()


    def __load_video_stream(self):

        # if a video path was not supplied, grab a reference to the webcam
        if self.input_path is None:
            print("[INFO] starting video stream...")
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            self.vs = cv2.VideoCapture(self.input_path)


    def __detect_faces(self):

        while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            frame = self.vs.read()
            frame = frame[1] if self.input_path is not None else frame

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if self.input_path is not None and frame is None:
                break

            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if self.output_path is not None and self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(self.output_path, fourcc, 30,(self.W, self.H), True)

            rects = []

            if self.totalFrames % self.skipframes == 0:
                # construct a blob from the frame, pass it through the network,
                # obtain our output predictions, and initialize the list of
                # bounding box rectangles
                blob = cv2.dnn.blobFromImage(frame, 1.0, (self.W, self.H), (104.0, 177.0, 123.0))
                self.face_detector_model.setInput(blob)
                detections = self.face_detector_model.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > self.confidence:

                        box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                        rects.append(box.astype("int"))
                        # draw a bounding box surrounding the object so we can
                        # visualize it
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        self.trackers.append(tracker)

            else:
                # loop over the trackers
                for tracker in self.trackers:
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

            objects = self.ct.update(rects)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = self.trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

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
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction_y > 0 and centroid[1] > int(self.H * 0.8):
                            self.totalvisitor += 1
                            to.counted = True

                # store the trackable object in our dictionary
                self.trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Total Visitors", self.totalvisitor),
                ("Total Tracked", len(self.trackers))
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if self.writer is not None:
                self.writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            self.totalFrames += 1
            self.fps.update()

        # stop the timer and display FPS information
        self.fps.stop()

        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # if we are not using a video file, stop the camera video stream
        if self.input_path is None:
            self.vs.stop()

        # otherwise, release the video file pointer
        else:
            self.vs.release()

        # close any open windows
        cv2.destroyAllWindows()

    def run(self):

        self.__load_video_stream()
        self.__detect_faces()







    