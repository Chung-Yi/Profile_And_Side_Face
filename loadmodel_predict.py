import cv2
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt
import efficientnet.keras
from operator import itemgetter
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical

model_name = 'models/81-0.01.h5'
IMG_SIZE = 100
image_list = glob.glob('images/front_profile_face_test/**/*.jpg')
model = load_model(model_name)

thrs = np.linspace(0, 1, 10)
print(len(image_list))


def swish_activation(x):
    return (K.sigmoid(x) * x)


def image_predict(image_list):
    ground_truth = []
    predicts = []
    images = []

    for image in image_list:
        # ground_truth_classes = int(image.split('/')[-1].split('_')[0])
        ground_truth_classes = int(image.split('/')[2])

        ground_truth.append(ground_truth_classes)

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[np.newaxis, :]
        img = img.astype("float32") / 255.0

        pred = model.predict(img)[0]
        predicts.append(pred)
        images.append(img)
    images = np.array(images)
    images = np.squeeze(images, axis=1)

    return ground_truth, predicts, images


def roc(ground_truth, predicts, thrs):
    total = len(ground_truth)
    tprs, fprs, accs, precisions, accuracy, f1_score = [], [], [], [], [], []
    pred = np.array(predicts)

    ground_truth = np.array(ground_truth)

    for thr in thrs:

        pred_front_index = np.where(pred[:, 1] > thr)[0]
        pred_side_index = np.where(pred[:, 1] <= thr)[0]
        tp = np.sum(
            list(map(lambda g: g == 1, ground_truth[pred_front_index])))

        fp = np.sum(
            list(map(lambda g: g == 0, ground_truth[pred_front_index])))

        tn = np.sum(list(map(lambda g: g == 0, ground_truth[pred_side_index])))

        fn = np.sum(list(map(lambda g: g == 1, ground_truth[pred_side_index])))

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (
            precision + tpr) != 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        accuracy.append(acc)
        f1_score.append(f1)

    print(tprs[np.argmax(accuracy)])
    print(fprs[np.argmax(accuracy)])

    recall = tprs

    return fprs, tprs, precisions, recall, accuracy, f1_score


def draw_roc_cure(fprs, tprs, area):

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fprs, tprs, label='AUC (area = {:.3f})'.format(area))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# def test_loss(ground_truth, predicts):
# matches = np.equal(ground_truth, np.argmax(predicts, 1))
# not_matches = np.not_equal(ground_truth, np.argmax(predicts, 1))
# acc = np.mean(np.cast[np.float32](matches))
# error = np.mean(np.cast[np.float32](not_matches))
# acc, loss = model.evaluate()


def area_calculation(x, y, n):

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_criteria(thrs, f1_score, accuracy, precisions, recall):
    criteria = max(f1_score)
    precision = precisions[np.argmax(f1_score)]
    recall = recall[np.argmax(f1_score)]
    acc = accuracy[np.argmax(f1_score)]
    thr = thrs[np.argmax(f1_score)]
    return acc, thr, precision, recall, criteria


def main():

    ground_truth, predicts, images = image_predict(image_list)
    loss, acc = model.evaluate(images, to_categorical(ground_truth, 2))
    print(loss, acc)

    fprs, tprs, precisions, recall, accuracy, f1_score = roc(
        ground_truth, predicts, thrs)

    # print(accuracy)
    # print(f1_score)

    acc, thr, precision, recall, f1_score = get_criteria(
        thrs, f1_score, accuracy, precisions, recall)
    print(f'accuracy:{round(acc, 2)}')
    print(f'threshold:{round(thr, 2)}')
    print(f'precision:{round(precision, 2)}')
    print(f'recall:{round(recall, 2)}')
    print(f'f1_score:{round(f1_score, 2)}')
    auc_keras = auc(fprs, tprs)
    draw_roc_cure(fprs, tprs, auc_keras)

    # area = area_calculation(fprs, tprs, len(fprs))
    # print(area + 0.5)
    # ='best')


if __name__ == '__main__':
    main()