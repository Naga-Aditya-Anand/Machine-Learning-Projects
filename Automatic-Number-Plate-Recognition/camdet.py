from ultralytics import YOLO
import cv2
import easyocr

bestmodel = YOLO('best.pt')

cam = cv2.VideoCapture(0)

#results[0].show()

reader = easyocr.Reader(['en'],gpu=False)

while True:
    ret,frame = cam.read()
    if not ret:
        break

    results = bestmodel.predict(frame,conf=0.25)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            crop_plate = frame[y1:y2,x1:x2]

            #preprocessing
            gray = cv2.cvtColor(crop_plate,cv2.COLOR_RGB2GRAY)
            _,thresh = cv2.threshold(gray,118,255,cv2.THRESH_BINARY)
            #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            #cv2.imwrite("thresh.jpg",thresh)
            ocr_res = reader.readtext(thresh)
            
            if ocr_res:
                number_plate_text = ocr_res[0][1]
                print("Number Plate Detected: ",number_plate_text)
                cv2.putText(frame,number_plate_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            else:
                print("Number Plate Text Not Recognized")

    # cv2.imshow("ANPR",frame)
    # cv2.waitKey(0)

# cam.release()
# cv2.destroyAllWindows()