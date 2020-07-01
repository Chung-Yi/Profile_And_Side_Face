import glob
import cv2
import dlib
import os
import efficientnet.keras
import numpy as np
import imutils
import math
import face_alignment
import numpy as np
from keras.models import load_model
from imutils import face_utils

# image_path = 'images/DAY1C/*.jpg'
image_path = 'images/front_profile_face_test_copy/*.jpg'
model_name = 'models/efficient_B5.h5'
front_path = 'images/front_profile_face_test_copy/front_face/'
profile_path = 'images/front_profile_face_test_copy/profile_face/'
save_image = 'images/face_landmarks/'
model = load_model(model_name)
IMG_SIZE = 100

thr = 0.22

model_name = "models/shape_predictor_81_face_landmarks.dat"


class PoseDetector:
    def __init__(self, face_image):
        self.face_image = face_image
        self.width = self.face_image.shape[1]
        self.height = self.face_image.shape[0]
        self.focal_length = self.width
        self.center = (self.width / 2, self.height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]], [0, 0, 1]],
            dtype="double")

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383],
                               [-2053.03596872]])

        # 3D model points
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ],
            dtype=np.float32)

    def detect_head_pose_with_6_points(self, points, name):

        # 2D image points
        image_points = np.array(
            [
                (points[30][0], points[30][1]),  # nose tip
                (points[8][0], points[8][1]),  # chin
                (points[36][0], points[36][1]),  # left eye
                (points[45][0], points[45][1]),  # right eye
                (points[48][0], points[48][1]),  # left mouth
                (points[54][0], points[54][1]),  # right mouth
            ],
            dtype=np.float32)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)

        # print("Rotation Vector:\n {0}".format(rotation_vector))
        # print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        (nose_end_point2D, jacobian1) = cv2.projectPoints(
            np.float32([[500, 0, 0], [0, 500, 0],
                        [0, 0, 500]]), rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeffs)

        (modelpts, jacobian2) = cv2.projectPoints(
            self.model_points, rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))

        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        # print(pitch, roll, yaw)

        for p in image_points:
            cv2.circle(self.face_image, (int(p[0]), int(p[1])), 2, (0, 0, 255),
                       -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))

        # cv2.line(self.face_image, p1, tuple(nose_end_point2D[1].ravel()),
        #          (0, 255, 0), 2)  #Green
        # cv2.line(self.face_image, p1, tuple(nose_end_point2D[0].ravel()),
        #          (255, 0, 0), 2)  #Blue
        # cv2.line(self.face_image, p1, tuple(nose_end_point2D[2].ravel()),
        #          (0, 0, 255), 2)  #RED

        # cv2.putText(
        #     self.face_image, ('pitch: {:05.2f}').format(float(str(pitch))),
        #     (10, 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (0, 255, 0),
        #     thickness=2,
        #     lineType=2)

        # cv2.putText(
        #     self.face_image, ('roll: {:05.2f}').format(float(str(roll))),
        #     (10, 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (255, 0, 0),
        #     thickness=2,
        #     lineType=2)

        # cv2.putText(
        #     self.face_image, ('yaw: {:05.2f}').format(float(str(yaw))),
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5, (0, 0, 255),
        #     thickness=2,
        #     lineType=2)

        # cv2.imwrite(save_image + name, self.face_image[..., ::-1])

        return yaw


def get_81_points(image, model_name):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_name)
    face_dets = detector(image, 0)

    for i, d in enumerate(face_dets):
        shape = predictor(image, d)
        landmarks = [[p.x, p.y] for p in shape.parts()]
        return landmarks


def face_landmark(face_image):

    points = get_81_points(face_image, model_name)
    if points == None:
        return None
    points = np.array(points).astype('float32')

    return points


def cnn_model():
    for image in glob.glob(image_path):
        name = image.split('/')[-1]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[np.newaxis, :]
        img = img.astype("float32") / 255.0

        pred = model.predict(img)[0]
        front_face_prb = pred[1]

        if not os.path.isdir(front_path):
            os.mkdir(front_path)
        if not os.path.isdir(profile_path):
            os.mkdir(profile_path)

        if front_face_prb >= thr:
            os.rename(image, front_path + name)
        else:
            os.rename(image, profile_path + name)


def landmarks_model():
    for img in glob.glob(image_path):
        name = img.split('/')[-1]
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pose = PoseDetector(image)
        pts = face_landmark(image)
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device='cpu', flip_input=True)
        try:
            pts = fa.get_landmarks(image)[-1]
        except:
            continue

        if pts is None:
            continue

        yaw_angle = pose.detect_head_pose_with_6_points(pts, name)

        if not os.path.isdir(front_path):
            os.mkdir(front_path)
        if not os.path.isdir(profile_path):
            os.mkdir(profile_path)

        if abs(yaw_angle) <= 50:
            os.rename(img, front_path + name)
        else:
            os.rename(img, profile_path + name)


def main():
    cnn_model()
    # landmarks_model()

    # image_name = 'D.jpg'
    # image = cv2.imread(image_name)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # pose = PoseDetector(image)
    # pts = face_landmark(image)
    # yaw_angle = pose.detect_head_pose_with_6_points(pts)

    # cv2.imwrite(image_name.split('.')[0] + '_points.jpg', image[..., ::-1])


if __name__ == '__main__':
    main()