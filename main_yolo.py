import cv2
import numpy as np


net=cv2.dnn.readNet("veri/yolov3-tiny.weights","veri/yolov3-tiny.cfg")


classes=[]
with open("veri/coco.names","r") as f:
    read=f.readlines()
for i in range(len(read)):
    classes.append(read[i].strip("\n"))



layer_names=net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])



img=cv2.imread("veri/img-1.jpg")
height,width,channels=img.shape


blob=cv2.dnn.blobFromImage(img,0.00392,(600,600),(0,0,0),True,crop=False)



net.setInput(blob)
outs=net.forward(output_layers)


class_ids=[]
confidences=[]
boxes=[]
for output in outs:
    for detection in output:

        scores=detection[5:]                
        class_id=np.argmax(scores)          
        confidence =scores[class_id]        

        if confidence >0.8: 
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)

         
            x=int(center_x-w/2) 
            y=int(center_y-h/2) 

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]  

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

cv2.imshow("Output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
