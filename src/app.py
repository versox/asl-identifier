import tkinter as tk
import tkinter.font as tkfont
import tkinter.filedialog as tkFileDialog
import cv2

from video import Video

def cameraBtnCallback():
    cI = int(camIndex.get())
    print(camIndex)
    v = Video.loadCam(cI)
    v.stream()
    del v

def fileCallback():
    filename = tkFileDialog.askopenfilename()
    v = Video.loadFile(filename)
    v.stream()
    del v

top = tk.Tk()

titleFont = tkfont.Font(size=24)
normalFont = tkfont.Font(size=14)

sourceSelect = tk.Frame(top)
sourceLabel = tk.Label(sourceSelect, text="Source", font=titleFont)
sourceLabel.pack(side=tk.TOP, fill=tk.X, pady=5)
tk.Label(sourceSelect, text="(use ESC to exit video)").pack()

cameraFrame = tk.Frame(sourceSelect) 
tk.Button(cameraFrame, command=cameraBtnCallback, text="Camera", font=normalFont,  background="lightblue").pack(side=tk.LEFT, padx=5)
camIndex = tk.Entry(cameraFrame, width=2, font=normalFont)
camIndex.insert(0, "0")
camIndex.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH)
cameraFrame.pack(pady=5)

fileFrame = tk.Frame(sourceSelect)
fileButton = tk.Button(fileFrame, command=fileCallback, text="File", font=normalFont, background="lightyellow").pack(fill=tk.X)
fileFrame.pack(pady=5)

sourceSelect.pack(pady=60, padx=40)


top.mainloop()