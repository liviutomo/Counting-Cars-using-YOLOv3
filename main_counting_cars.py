# importam librariile necesare
import numpy as np
import imutils
import time
import cv2
import os
import glob

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)
from sort import *

tracker = Sort()
memory = {}
#line = [(44, 543), (550, 655)]
# for video highway_traffic
#line = [(278, 424), (600, 434)]
#line2 = [(666, 429), (955, 432)]

# for video test2
# line = [(x1, y1), (x2, y2)]
#line = [(264, 175), (264, 290)]
#line2 = [(388, 270), (438, 476)]

#for video Test3
#line = [(278, 424), (955, 432)]

#for video Test3_crop
line = [(113, 277), (595, 277)]
counter = 0
counter2 = 0



# Returneaza true daca liniile de segment AB si CD se intersectează
# Adica p0  p1 cu line[0] line[1]
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# importa fisierul cu numele claselor pe care a fost antrenat modelul
# nostru de retea YOLO

with open('yolo-coco-data/classes.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]
# initialize a list of colors to represent each possible class label
# initializam o lista de culori care sa reprezinte eticheta
# fiecarei clase. aceasta linie de cod este utila in caz ca
# aplicatia se dezvolta detectand un numar mai mare de clase
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# Se importa reteaua YOLO impreuna cu ponderile aferente
# se determina doar numele stratului de iesire de care avem nevoie din YOLO
print("[INFO] Se importa YOLO cu ponderile aferente...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3_test.cfg',
                                 'yolo-coco-data/yolov3_train_final_50kpoze.weights')
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# este initializat fisierul video
#vs = cv2.VideoCapture(args["input"])
vs = cv2.VideoCapture("input/highway_crop.mp4")
writer = None
(W, H) = (None, None)
confidence = 0.5
threshold = 0.3
frameIndex = 0

# se determina numarul total de frame-uri
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] Sunt {} de cadre in video".format(total))

# exceptie in caz ca nu se poate determina numarul de cadre

except:
    print("[INFO] Nu s-a putut determina numarul de cadre.")
    print("[INFO] Nu se poate realiza o aproximatie in timp.")
    total = -1

# loop over frames from the video file stream
# bucla while care cuprinde toate cadrele
while True:
    # Se citeste urmatorul frame din fisier
    (grabbed, frame) = vs.read()

    # daca nu a fost gasit nici un frame, grabbed
    # atunci am ajuns la sfarsitul videoului

    if not grabbed:
        break
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    # if the frame dimensions are empty, grab them

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    #
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # lista cu casetele de delimitare detectate,
    # gradul de incredere, clasa

    boxes = []
    confidences = []
    classIDs = []

    # loop peste fiecare strat de iesire
    for output in layerOutputs:
        # loop peste fiecare detectie
        for detection in output:
            # se extrag id-ul clasei si gradul de incredere
            # al obiectului curent detectat
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # sunt filtrate obiectele detectate care au un grad
            # de incredere mai mic decat pragul setat
            if confidence > 0.5:
                # sunt scalate casetele de delimitare pentru a incadra
                # detectia. YOLO mentine centrul coordonatelor casetelor
                # de delimitare (x, y) si mai avem nevoie doar de latime
                # si inaltime
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # foloseste centrele (x, y) pentru a determina
                # coltul de sus si cel din stanga al casetei
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # sunt actualizate listele de parametrii si coordonatele
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # se aplica non-maxima suppression pentru a elimina casetele slabe
    # si cele care se suprapun
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    dets = []
    if len(idxs) > 0:
        # loop peste lista suprimată idxs
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    # este creat un vector pe baza noii liste de detecții.
    # Fiecare detecție este supusă algoritmului de urmarire
    # SORT.
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}
    # se creaza lista memory, care va actualiza previous

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # se extrag coordonatele casetei de delimitare
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            # se creaza casetele anterioare si se calculeaza
            # centrul casetei
            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                cv2.line(frame, p0, p1, color, 3)

                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                #if intersect(p0, p1, line2[0], line2[1]):
                 #   counter2 += 1

            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # deseneaza linia
    cv2.line(frame, line[0], line[1], (0, 255, 255), 10)
    #cv2.line(frame, line2[0], line2[1], (0, 255, 255), 5)
    # scrie contorul
    text_in = 'IN: ' + str(counter)
    #text_out = 'OUT:' + str(counter2)
    #for video highway_traffic
    #cv2.putText(frame, text_in, (100, 200), cv2.FONT_HERSHEY_TRIPLEX, 5.0, (0, 0, 0), 5)
    #cv2.putText(frame, text_out, (750, 200), cv2.FONT_HERSHEY_TRIPLEX, 5.0, (0, 0, 0), 5)

    #for video test2
    #cv2.putText(frame, text_in, (150, 135), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 0, 0), 2)
    #cv2.putText(frame, text_out, (450, 135), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 0, 0), 2)
    # counter += 1
    #For video Test3
    #cv2.putText(frame, text_in, (100, 200), cv2.FONT_HERSHEY_TRIPLEX, 5.0, (0, 0, 0), 5)
    cv2.putText(frame, text_in, (51, 51), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 5)
    #For video Test3_crop

    # saves image file
    cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('output/results.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # sunt scrise frameurile in locatia output
    writer.write(frame)

    # urmatorul frame
    frameIndex += 1


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
