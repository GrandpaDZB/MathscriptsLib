import tensorflow_hub as hub
import cv2
import numpy as np
import tensorflow as tf

detector = hub.load("/home/grandpadzb/tfhub_modules/ssd_mobilenet_v2_2")
print("Complete loading")


camera = cv2.VideoCapture(0)

while(cv2.waitKey(1) != 113):
    _,src = camera.read()
    # src = cv2.imread("/home/grandpadzb/MathscriptsLib/selfDifineNetwork/party.jpg")
    src = cv2.resize(src, dsize=(320,320))
    src = src[np.newaxis,:]
    img = tf.convert_to_tensor(src, dtype="uint8")

    # ==========================
    output = detector(img)
    figure_num = 0
    for i in range(int(output["num_detections"].numpy()[0])):
        class_index = output["detection_classes"].numpy()[0][i]
        if class_index == 1.0:
            box = np.fix(output["detection_boxes"].numpy()[0][i]*320)
            cv2.rectangle(src[0,:], (box[1],box[0]), (box[3],box[2]), (255,255,255), 2)
            figure_num += 1
        if figure_num >= 1:
            break
    cv2.imshow("result", src[0,:])
    src = src[0,:]
    # cv2.waitKey(0)
cv2.destroyAllWindows()
camera.release()











