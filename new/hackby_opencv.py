import numpy as np
import cv2

cap = cv2.VideoCapture('AviFileChunk1_View3.mov')
cap.get(cv2.CAP_PROP_FRAME_COUNT)

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

print cap.get(3), cap.get(4), cap.get(cv2.CAP_PROP_FRAME_COUNT)

cap.release()
cv2.destroyAllWindows()