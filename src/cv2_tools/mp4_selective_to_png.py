import cv2


folder = "./thunderhill/run_4/"
mov = "DJI_0007"
filepath = folder+mov+".MOV"


def video_to_mp4(input, output, fps: int = 0, frame_size: tuple = (), fourcc: str = "H264"):
    vidcap = cv2.VideoCapture(input)
    if not fps:
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    success, arr = vidcap.read()
    if not frame_size:
        height, width, _ = arr.shape
        frame_size = width, height
    writer = cv2.VideoWriter(
        output,
        apiPreference=0,
        fourcc=cv2.VideoWriter_fourcc(*fourcc),
        fps=fps,
        frameSize=frame_size
    )
    while True:
        if not success:
            break
        writer.write(arr)
        success, arr = vidcap.read()
    writer.release()
    vidcap.release()

vidcap = cv2.VideoCapture(filepath)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)

    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(folder +mov+"/extras" + "/image_"+ str(count)+".png", image)     # save frame as JPG file
    return hasFrames
sec = 16
frameRate = 0.1 #//it will capture image in each 0.5 second
count=1
max_count = 30
success = getFrame(sec)
print(success)
while success and count<max_count:
    print(count)
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print("Done")