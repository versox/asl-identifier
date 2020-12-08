import cv2
import mediapipe as mp
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# need to get binary file from whatsapp and put it in appropiate directory
# if using a CONDA environment:
#   (for me it was)  C:\Users\Carson\miniconda3\envs\aslIdentify\Lib\site-packages\mediapipe\modules\palm_detection
# otherwise it will be in the python install location
# I think it will log where it is supposed to be in the error message when you try running this
BIN_FILEPATH = 'mediapipe/modules/palm_detection/palm_detection_cpu.binarypb'

handBox = mp.solution_base.SolutionBase(binary_graph_path=BIN_FILEPATH)

top = 250
left = 200
right = 200
bottom = 150

# Returns a tuple   (view, crop)
# where view is an image with the bounding boxes drawn
# and crop is an image crop of the bounding box
def getBounding(image):
    inputX = image.shape[1]
    inputY = image.shape[0]

    results = handBox.process(image[:, :, ::-1])

    view = image
    crops = None
    if (results.detections):
        crops = [None] * len(results.detections) # init empty arr
        for i, d in enumerate(results.detections):
            boundingBox = d.location_data.relative_bounding_box
            # Original bounding box
            # x1 = int(boundingBox.xmin * inputX)
            # y1 = int(boundingBox.ymin * inputY)
            # x2 = int(x1 + (boundingBox.width * inputX))
            # y2 = int(y1 + (boundingBox.height * inputY))
            # view = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Stretched
            x1 = int((boundingBox.xmin - 0.2) * inputX)
            y1 = int((boundingBox.ymin - 0.3)  * inputY)
            x2 = int(boundingBox.xmin * inputX + ((boundingBox.width + 0.2) * inputX))
            y2 = int(boundingBox.ymin * inputY + ((boundingBox.height + 0.1) * inputY))
            if (x1 < 0):
                x1 = 0
            if (y1 < 0):
                y1 = 0
            if (x2 >= inputX):
                x2 = inputX - 1
            if (y2 >= inputY):
                y2 = inputY - 1
            crops[i] = image[y1:y2, x1:x2].copy()
            view = cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # print('X1:', x1, 'X2:', x2, 'Y1:', y1, 'Y2:', y2)

    return (view, crops)