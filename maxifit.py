import numpy as np
import cv2
middle_cascade = cv2.CascadeClassifier("/home/hp/PycharmProjects/tryOn/haarcascades/haarcascade_upperbody.xml")
face_cascade=cv2.CascadeClassifier("/home/hp/PycharmProjects/tryOn/haarcascades/haarcascade_frontalface_default.xml")


dress = cv2.imread("/home/hp/Downloads/dataset/dress(western)/Formal/gown17.png", cv2.IMREAD_UNCHANGED)
bgr = dress[:,:,:3]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Some sort of processing...
fw=0

#bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
alpha = dress[:,:,3] # Channel 3
result = np.dstack([bgr, alpha])
result = cv2.resize(result,(300,450))
cv2.imshow("dress",dress)
girl = cv2.imread("/home/hp/girl/finalg.jpg")
girl=cv2.resize(girl,(289,385))
gray = cv2.cvtColor(girl, cv2.COLOR_BGR2GRAY)  # convert video to grayscale
Fh = 0
Fw = 0
Fy = 0
face = face_cascade.detectMultiScale(girl, 1.1, 4)
middle_body = middle_cascade.detectMultiScale(
    gray,
    scaleFactor=1.01,
    minNeighbors=1,
    minSize=(1, 1),  # Min size for valid detection, changes according to video size or body size in the video.
    flags=cv2.CASCADE_SCALE_IMAGE
)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found,w = hog.detectMultiScale(girl, winStride=(8,8), padding=(32,32), scale=1.05)
Fy = 0
Fw = 0
for (fx, fy, fw, fh) in face:
    Fy=fh
    #cv2.rectangle(girl, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 6)
for (mx,my,mw,mh) in  middle_body:
    #cv2.rectangle(girl, (mx, my), (mx + mw, my + mh), (0, 255, 0), 6)
    Fw=mw
for x, y, w, h in found:
    # the HOG detector returns slightly larger rectangles than the real objes.
    # so we slightly shrink the rectangles to get a nicer output.
    pad_w, pad_h = int(0.15 * w), int(0.05 * h)
    #cv2.rectangle(girl, (x, y), (x + w, y + h), (0, 255, 0), 10)
    girl = cv2.cvtColor(girl, cv2.COLOR_BGR2BGRA)
    dress = cv2.resize(result, (w +Fw-50, h-25))
    # girl=cv2.resize(girl,(fw,girl.shape[0]))

    w, h, c = dress.shape
    for i in range(0, w):
        for j in range(0, h):

            if dress[i, j][3] != 0:
                girl[y + i+25, x + j-Fw+83] = dress[i, j]
cv2.imshow("girl",girl)
cv2.waitKey(0)
