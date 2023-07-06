import cv2
import numpy as np
from sklearn.cluster import KMeans

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        reshaped_roi = roi_color.reshape((-1, 3)) 
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(reshaped_roi)
        segmented_roi = kmeans.cluster_centers_[kmeans.labels_] 
        
        segmented_roi = segmented_roi.reshape(roi_color.shape).astype(np.uint8)
        
        cv2.imshow('Segmentacao de Rosto', segmented_roi)
    
    cv2.imshow('Deteccao Facial', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
