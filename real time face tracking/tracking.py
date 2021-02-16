import cv2

cap = cv2.VideoCapture(0)
# load cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# declare tracker
tracker = cv2.TrackerMedianFlow_create()
# declare variable for checking, Is it tracking or not.
#starting with False
onTracking = False

while True:
    ret, frame = cap.read()
    # if onTracking is false : let do face detection
    if not onTracking:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detecMultiScale(gray, 1.1, 4)
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # if track initial success : let onTracking is True
            if tracker.init(frame, (x,y,w,h)):
                onTracking = True
    # if onTracking is true : let do update tracking
    else:
        # update in next frame (track the moving face)
        check, boundingbox = tracker.update(frame)
        if check:
            p1 = (int(boundingbox[0]), int(boundingbox[1]))
            p2 = (int(boundingbox[0]+boundingbox[2]), int(boundingbox[1]+boundingbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2)
        else: # in case track is out bound
             onTracking = False
             # create tracking again 
             tracker = cv2.TrackerMedianFlow_create() 
    # output
    cv2.imshow('frame', frame)
    cv2.waitKey(1)