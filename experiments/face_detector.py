# import the necessary packages
# from utils.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
args = vars(ap.parse_args())

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
MIN_CONFIDECE = 0.65

# initialize our centroid tracker and frame dimensions
# ct = CentroidTracker(maxDisappeared=30)
trackers = {}
tracker = None
(H, W) = (None, None)
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('../models/face_detector.prototxt','models/face_detector.caffemodel')

print("[INFO] loading age detector model...")
ageNet = cv2.dnn.readNet('../models/age_detector.prototxt', 'models/age_detector.caffemodel')

# initialize the video stream and allow the camera sensor to warmup
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# loop over the frames from the video stream

writer = None
totalFrames = 0
skip_frames = 2

while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=800)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if args["input"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('../samples/output.avi', fourcc, 30,
			(W, H), True)


	if tracker is None:
		# construct a blob from the frame, pass it through the network,
		# obtain our output predictions, and initialize the list of
		# bounding box rectangles
		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# filter out weak detections by ensuring the predicted
			# probability is greater than a minimum threshold
			if detections[0, 0, i, 2] > MIN_CONFIDECE:
				# compute the (x, y)-coordinates of the bounding box for
				# the object, then update the bounding box rectangles list
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				face = frame[startY:endY, startX:endX]
				# ensure the face ROI is sufficiently large
				if face.shape[0] < 20 or face.shape[1] < 20:
					continue

				faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

				# make predictions on the age and find the age bucket with
				# the largest corresponding probability
				ageNet.setInput(faceBlob)
				preds = ageNet.forward()
				i = preds[0].argmax()
				age = AGE_BUCKETS[i]
				ageConfidence = preds[0][i]

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# draw a bounding box surrounding the object so we can
				# visualize it
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
				cv2.putText(frame, age, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	else:
		# update the tracker and grab the position of the tracked
		# object
		tracker.update(rgb)
		pos = tracker.get_position()
		# unpack the position object
		startX = int(pos.left())
		startY = int(pos.top())
		endX = int(pos.right())
		endY = int(pos.bottom())
		# draw the bounding box from the correlation object tracker
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.putText(frame, age, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	totalFrames += 1

if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# do a bit of cleanup
cv2.destroyAllWindows()