import face_alignment
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections

image_list = glob.glob('images/DAY1B/index_20/**.jpg')
image_list = glob.glob('front.jpg')
save_image = 'images/face_landmarks/'


def area_calculation(x, y):

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def face_remap_points(shape):

    remapped_image = cv2.convexHull(shape)

    return remapped_image


def crop_left_face(preds, input_img):
    left_face = np.concatenate((preds[:9], preds[27:31], preds[17:22]), axis=0)
    points_int = np.array([[int(p[0]), int(p[1])] for p in left_face])
    remapped_shape = np.zeros_like(left_face)
    left_landmark_face = np.zeros_like(input_img)
    feature_mask = np.zeros((input_img.shape[0], input_img.shape[1]))

    remapped_shape = face_remap_points(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[:], 1)
    feature_mask = feature_mask.astype(np.bool)
    left_landmark_face[feature_mask] = input_img[feature_mask]

    return left_landmark_face


def crop_right_face(preds, input_img):
    right_face = np.concatenate((preds[8:17], preds[27:31], preds[22:27]),
                                axis=0)
    points_int = np.array([[int(p[0]), int(p[1])] for p in right_face])
    remapped_shape = np.zeros_like(right_face)
    right_landmark_face = np.zeros_like(input_img)
    feature_mask = np.zeros((input_img.shape[0], input_img.shape[1]))

    remapped_shape = face_remap_points(points_int)

    cv2.fillConvexPoly(feature_mask, remapped_shape[:], 1)
    feature_mask = feature_mask.astype(np.bool)
    right_landmark_face[feature_mask] = input_img[feature_mask]

    return right_landmark_face


def main():
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device='cpu', flip_input=True)

    for image in image_list:

        input_img = cv2.imread(image)

        try:
            preds = fa.get_landmarks(input_img)[-1]
        except:
            continue

        # 2D-Plot
        plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)

        pred_type = collections.namedtuple('prediction_type',
                                           ['slice', 'color'])
        pred_types = {
            'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
        }

        left_face = crop_left_face(preds, input_img)
        right_face = crop_right_face(preds, input_img)

        left_face_points = np.concatenate(
            (preds[:9], preds[27:31], preds[17:22]), axis=0)
        right_face_points = np.concatenate(
            (preds[8:17], preds[27:31], preds[22:27]), axis=0)

        left_area = area_calculation(left_face_points[:, 0],
                                     left_face_points[:, 1])
        right_area = area_calculation(right_face_points[:, 0],
                                      right_face_points[:, 1])

        print(left_area, right_area)

        # print(preds[:, 0])

        # for pred_type in pred_types.values():

        #     for x, y in zip(preds[pred_type.slice, 0],
        #                     preds[pred_type.slice, 1]):
        #         cv2.circle(input_img, (x, y), 2, (255, 0, 0), -1)

        # cv2.imwrite(save_image + image.split('/')[-1], input_img)

        cv2.imshow('My Image', right_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # # ax.axis('off')

        # # 3D-Plot
        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        # surf = ax.scatter(
        #     preds[:, 0] * 1.2,
        #     preds[:, 1],
        #     preds[:, 2],
        #     c='cyan',
        #     alpha=1.0,
        #     edgecolor='b')

        # for pred_type in pred_types.values():
        #     ax.plot3D(
        #         preds[pred_type.slice, 0] * 1.2,
        #         preds[pred_type.slice, 1],
        #         preds[pred_type.slice, 2],
        #         color='blue')

        # ax.view_init(elev=90., azim=90.)
        # ax.set_xlim(ax.get_xlim()[::-1])
        # plt.show()


if __name__ == "__main__":
    main()