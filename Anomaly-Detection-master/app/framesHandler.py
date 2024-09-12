import os
from os.path import isfile, join
import time
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

path = 'app/static/anoframes'
count = 1
frames = []

def email():
    print("Sending email")
    HOST_ADDRESS = 'smtp.gmail.com'
    HOST_PORT = 587
    MY_ADDRESS = "apartment.security69@cumminscollege.in"     
    MY_PASSWORD = "security69"     
    RECIPIENT_ADDRESS = "apartment.security69@cumminscollege.in"  
    server = smtplib.SMTP(host=HOST_ADDRESS, port=HOST_PORT)
    server.starttls()
    server.login(MY_ADDRESS, MY_PASSWORD)
    message = MIMEMultipart()
    message['From'] = MY_ADDRESS
    message['To'] = RECIPIENT_ADDRESS
    message['Subject'] = "Activity Alert"
    textPart = MIMEText("Suspicious Activity identified. Please check. \n\nFrom Group No.69 ", 'plain')
    message.attach(textPart)
    server.send_message(message)
    server.quit()
'''
def createVid():
    global count
    print("Creating Video")
    #frameSize = (640, 480)
    frameSize = (800, 600)
    
    frames = os.listdir(path)
    name = frames[0].split('_')[0]
        
    out = cv2.VideoWriter(os.path.join('app/static/vids/',name+'{}.webm'.format(count)),cv2.VideoWriter_fourcc(*'VP80'), 3, frameSize)

    for filename in sorted(glob.glob('app/static/anoframes/*.jpg'),key=os.path.getmtime):
        img = cv2.imread(filename)
        out.write(img)
        print(filename+"written")

    out.release()
    count+=1
'''

def createVid():
    global count
    
    pathIn= 'app/static/anoframes/'
    
    fps = 5.0

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    name = files[0].split('_')[0]
    pathOut = os.path.join('app/static/vids/',name+str(count)+'.webm')
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'VP80'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def delFrames():
    print("Deleting Frames")
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


while True:
    dir = os.listdir(path)
    if len(dir)!=0:
        print("Starting 2 min counter")
        time.sleep(60*2)
        createVid()
        time.sleep(2)
        email()
        delFrames()
        print("Deleted")
        print(dir)

    