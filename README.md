# Realtime People Counter

Sample video taken from HIKvision demo of people counting product in Youtube.
My result is the red colored ones.
![Result Preview](https://github.com/astandri/realtime-people-counter/blob/master/demo.png)

## Features
Can capture Videos from:
- Video files
- IP camera (RTSP)
- Webcam
- Raspberry pi Camera

## System Overview
There will be 2 parts of the overall system
1. Client --> Reading frames from input source (camera, video files, etc)
2. Processor --> Receiving frames and write the output as a video stream

The output can be consumed using a direct link to the flask endpoint.
