import dlib
import cv2
import imutils
import os
import glob
import numpy as np
from imutils import face_utils

model_name = "models/shape_predictor_68_face_landmarks.dat"
folder1 = "DAY2C"
folder2 = "DAY2C_landmarks"
folder3 = "index_2"
image_path = f"images/{folder1}/{folder3}/*.jpg"
save_path = f"images/{folder1}/{folder2}/{folder3}"

THR = 35


def main():

    if not os.path.isdir(f"images/{folder1}/{folder2}"):
        os.mkdir(f"images/{folder1}/{folder2}")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not os.path.isdir(os.path.join(save_path, "0")):
        os.mkdir(os.path.join(save_path, "0"))

    if not os.path.isdir(os.path.join(save_path, "1")):
        os.mkdir(os.path.join(save_path, "1"))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_name)
    images = glob.glob(image_path)

    for image_name in images:
        name = image_name.split("/")[-1]

        image = cv2.imread(image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)

            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(shape) != 68:
                break
            nose_x = shape[33][0]
            nose_y = shape[33][1]

            right_cheek_x = shape[15][0]
            right_cheek_y = shape[15][1]

            left_cheek_x = shape[1][0]
            left_cheek_y = shape[1][1]

            right_dis = abs(nose_x - right_cheek_x)
            left_dis = abs(nose_x - left_cheek_x)

            delta_dis = abs(right_dis - left_dis)

            # for (x, y) in shape:
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            # cv2.line(image, (nose_x, nose_y), (right_cheek_x, nose_y),
            #          (0, 0, 255), 5)

            # cv2.putText(image, str(round(delta_dis)), (10, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
            #             cv2.LINE_AA)

            # cv2.imwrite(os.path.join(save_path, name), image)

            try:
                if delta_dis < THR:
                    os.rename(image_name, os.path.join(save_path, "1", name))
                else:
                    os.rename(image_name, os.path.join(save_path, "0", name))
            except:
                print(f"{image_name} has more than one face")

    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()