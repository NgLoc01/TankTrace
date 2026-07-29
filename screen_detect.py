import glob
import os
import tkinter

import cv2
import mss
import numpy as np
from ultralytics import YOLO


#Find the most recently trained model (glob = global)
candidates = glob.glob(os.path.join('.', 'runs', 'detect', 'train*', 'weights', 'best.pt'))  #all train*/weights/best.pt paths

model_path = max(candidates, key=os.path.getmtime)  #most recently modified file out of the candidates list

#Load the trained model and set up detection settings
model = YOLO(model_path)                #loads the newest trained weights found earlier
threshold = 0.5                         #minimum confidence score to draw a detection
class_name_dict = {0: 'sherman_tank'}   #maps class id -> label text

#splitting each side's own screen size in half keeps them from overlapping
_root = tkinter.Tk()
_root.withdraw()
screen_w, screen_h = _root.winfo_screenwidth(), _root.winfo_screenheight()  #logical size, same units cv2 uses to position windows
_root.destroy()

with mss.mss() as sct:
    monitor = sct.monitors[1]  #primary monitor, in whatever units mss captures with

#Capture the left half of the screen 
capture_region = {"left": monitor["left"], "top": monitor["top"],
                   "width": monitor["width"] // 2, "height": monitor["height"]}

#Preview window goes in the right half, so it's never inside capture_region and can't capture itself
preview_x, preview_w = screen_w // 2, screen_w // 2
preview_h = screen_h

#Continuously grab screenshots, run detection for each frame
try:
    with mss.mss() as sct: #opens a screen-capture session using the mss
        cv2.namedWindow('Tank Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tank Detection', preview_w, preview_h)
        cv2.moveWindow('Tank Detection', preview_x, 0)

        while True:
            screenshot = np.array(sct.grab(capture_region))         #grab the screen region as an image
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)    #reformat to the format the YOLO model expects

            results = model(frame)[0] #run the model on this frame

            for result in results.boxes.data.tolist(): #loop over every object detected in this frame (can be several sherman tanks in the frame)
                x1, y1, x2, y2, score, class_id = result

                if score > threshold: #only draw detections the model is confident enough about
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  #draw the box
                    cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)       #label the box

            cv2.imshow('Tank Detection', frame) #show this frame (with any boxes/labels drawn)

            if cv2.waitKey(1) & 0xFF == ord('q'):  #'q' key quits
                break
            if cv2.getWindowProperty('Tank Detection', cv2.WND_PROP_VISIBLE) < 1:  #window closed manually
                break
except KeyboardInterrupt:
    pass

cv2.destroyAllWindows() #close the preview window once the loop ends
