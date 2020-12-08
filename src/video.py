import cv2
import mpHelper

class Video():
    __createKey = object()

    @classmethod
    def loadCam(cls, camIndex: int):
        cap = cv2.VideoCapture(camIndex)
        if not cap.isOpened():
            raise IOError("can't open webcam!")
        return Video(cls.__createKey, True, cap)

    @classmethod
    def loadFile(cls, filePath: str):
        cap = cv2.VideoCapture(filePath)
        if not cap.isOpened():
            raise IOError("can't open file")
        return Video(cls.__createKey, False, cap)

    # private constructor
    def __init__(self, createKey, cam: bool, cap: cv2.VideoCapture):
        assert(createKey == Video.__createKey), \
            "Use loadCam or loadFile"
        self.cam = cam
        self.cap = cap

    def __del__(self):
        self.cap.release()
        cv2.destroyWindow(str(id(self)))

    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: # EOF
                break
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # No crop with box
            # cv2.imshow(str(id(self)), mpHelper.getBounding(frame)[0])
            
    # Crop
            crops = mpHelper.getBounding(frame)[1]
            if (crops and len(crops) > 0):
                cv2.imshow(str(id(self)), crops[0])

            c = cv2.waitKey(1)
            if c == 27: #ESC key
                break