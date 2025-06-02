from ultralytics import YOLO
import cv2
import easyocr

bestmodel = YOLO('best.pt')

results = bestmodel.predict('car3.jpeg')

#results[0].show()

reader = easyocr.Reader(['en'],gpu=False)

image = cv2.imread('car3.jpeg')
img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

for r in results:
    for box in r.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(0,255,0),2)

        crop_plate = img_rgb[y1:y2,x1:x2]

        #preprocessing
        gray = cv2.cvtColor(crop_plate,cv2.COLOR_RGB2GRAY)
        #_,thresh = cv2.threshold(gray,118,255,cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        cv2.imwrite("thresh.jpg",thresh)
        ocr_res = reader.readtext(thresh)

        if ocr_res:
            number_plate_text = ocr_res[0][1]
            print("Number Plate Detected: ",number_plate_text)
            cv2.putText(img_rgb,number_plate_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        else:
            print("Number Plate Text Not Recognized")

cv2.imshow("Image",img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()