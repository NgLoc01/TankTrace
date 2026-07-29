import glob
import os
import subprocess
import time
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# Builds and set the correct paths for output video
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'tank1.mp4')
video_path_out = '{}_trained_out.mp4'.format(video_path)
video_path_silent = '{}_trained_out.silent.mp4'.format(video_path)

#Find the most recently trained model (glob = global)
candidates = glob.glob(os.path.join('.', 'runs', 'detect', 'train*', 'weights', 'best.pt'))  #all train*/weights/best.pt paths
model_path = max(candidates, key=os.path.getmtime)  #most recently modified file out of the candidates list

#Open the video file and grab its first frame
cap = cv2.VideoCapture(video_path)  #opens tank video tank1.mp4 as a video stream and returns a capture object (cap) you can read frames from
ret, frame = cap.read()             #reads the next frame from that stream.

#Set up the output video writer to match the input video's size and speed
H, W, _ = frame.shape            #frame size
fps = cap.get(cv2.CAP_PROP_FPS)  #frame rate
out = cv2.VideoWriter(video_path_silent, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H))  #writer for the silent annotated video

# Load the trained model and set up detection settings
model = YOLO(model_path)                #loads the newest trained weights found earlier
threshold = 0.5                         #minimum confidence score to draw a detection
class_name_dict = {0: 'sherman_tank'}   #maps class id -> label text

#Track progress while looping through every frame of the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress = tqdm(total=total_frames, desc="Detecting (trained)", unit="frame")
start_time = time.time()

#Run detection on every frame, draw boxes above the threshold, then write the frame out
while ret:

    results = model(frame, verbose=False)[0]  #run the model on this frame

    for result in results.boxes.data.tolist():  #loop over every object detected in this frame (can be several tanks in the frame)
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:  #only draw detections the model is confident enough about
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)  #draw the box
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  #label the box

    out.write(frame)          #save this annotated frame to the silent output video
    progress.update(1)
    ret, frame = cap.read()   #grab the next frame (ret becomes False at end of video)

progress.close()
detect_elapsed = time.time() - start_time
cap.release()
out.release()

#Mux (combining separate audio and video streams into one file) the original audio back onto the silent annotated video, then clean up the silent file
subprocess.run([
    'ffmpeg', '-y',
    '-i', video_path_silent,
    '-i', video_path,
    '-map', '0:v:0', '-map', '1:a:0',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
    '-shortest',
    video_path_out,
], check=True)
os.remove(video_path_silent)

#Report total runtime and where the final output video was saved
total_elapsed = time.time() - start_time
print(f"Done. Output saved to {video_path_out}")
print(f"Detection took {detect_elapsed:.1f}s, total (incl. muxing) took {total_elapsed:.1f}s")

#takes about 14 min to run