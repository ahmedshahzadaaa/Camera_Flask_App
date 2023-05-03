from flask import *
import cv2
import datetime
import time
import os
import numpy as np
import pyaudio
from PIL import Image
from werkzeug import *
from threading import *





global capture, rec_frame, grey, switch, neg, face, rec, out,menu,logo

capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
menu =0
logo =0


# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instantiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if face:
                frame = detect_face(frame)
            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if rec:
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def index1():
    return render_template('index1.html')




@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face

            if face:
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if switch == 1:
                switch = 0
                camera.release()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()

            elif not rec:
                out.release()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


# for image
@app.route('/request', methods=['POST', 'GET'])
def display():
    if request.method == 'POST':
        if request.form.get('menu') == 'menu':
            global menu
            im=Image.open('templates/menu.jpg')
            im.show()
            menu = not menu
            if request.form.get('menu') == 'menu':
                im.close()

        elif request.form.get('logo') == 'logo':
            global logo
            # read the image
            im = Image.open('templates/logo.jpg')
            # show image
            im.show()
            logo = not logo
            if request.form.get('logo') == 'logo':
                im.close()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

# end display



# for mic stream
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 0
AUDIO_DELAY: -3.5

audio1 = pyaudio.PyAudio()


def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000 * 10 ** 6
    o = bytes("RIFF", 'ascii')  # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4, 'little')  # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE", 'ascii')  # (4byte) File type
    o += bytes("fmt ", 'ascii')  # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, 'little')  # (4byte) Length of above format data
    o += (1).to_bytes(2, 'little')  # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2, 'little')  # (2byte)
    o += (sampleRate).to_bytes(4, 'little')  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, 'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2, 'little')  # (2byte)
    o += (bitsPerSample).to_bytes(2, 'little')  # (2byte)
    o += bytes("data", 'ascii')  # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4, 'little')  # (4byte) Data size in bytes
    return o


@app.route('/audio')
def audio():

    def sound():

        CHUNK = 1024
        sampleRate = 44100
        bitsPerSample = 16
        channels = 2
        AUDIO_DELAY: -3.5
        wav_header = genHeader(sampleRate, bitsPerSample, channels)

        stream = audio1.open(format=FORMAT, channels=CHANNELS,
                             rate=RATE, input=True, input_device_index=1,
                             frames_per_buffer=CHUNK)
        # frames = []
        first_run = True
        while True:
            if first_run:
                data = wav_header + stream.read(CHUNK)
                first_run = False
            else:
                data = stream.read(CHUNK)
            yield (data)

    return Response(sound())

def audio_index():
    """Video streaming home page."""
    return render_template('index.html')
#stop

# #out put mic
# if request.method == 'POST':
#     try:
#         target =  request.form["target"]
#         try:
#             samplerate = 16000
#             file =  request.files["file"]
#             x = ''.join(random.choice(string.ascii_lowercase) for i in range(6))
#             src = f'{temp_folder}/{x}.wav' #file.name
#             data, samplerate = sf.read(io.BytesIO(file.read()))
#             print("\n\n\n\n")
#             print("*"*30)
#             print(len(data), samplerate)
#             sf.write(src,data, samplerate)

# # end

@app.route('/audiorecog', methods = ['GET', 'POST'])
def audiorecog(video_stream=None):
   if request.method == 'POST':
      print("Recieved Audio File")
      file = request.files['file']
      print('File from the POST request is: {}'.format(file))
      with open("audio.wav", "wb") as aud:
            aud_stream = file.read()
            aud.write(video_stream)
      return "Success"
   return 'Call from get'

""


import ssl
if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5031, debug=True, ssl_context=('cert.pem', 'key.pem'))
    https = False
    https =  True
    mainpath = os.path.abspath(os.getcwd())
    if https==True:
        context = ssl.SSLContext()
        context.load_cert_chain('cert.pem','key.pem')
        app.run(host='0.0.0.0', port=5031, debug=False , ssl_context=context)
    else:
        app.run(host='0.0.0.0', port=5031, debug=False)
    
    
    
camera.release()

