import pickle
import numpy as np
import math
import cv2
import matplotlib
from matplotlib import pyplot as plt

eps = 0.01


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks in all_hand_peaks:
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        peaks = np.array(peaks)
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)
    return canvas


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            X = candidate[index.astype(int), 0] * float(W)
            Y = candidate[index.astype(int), 1] * float(H)
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    return canvas


def draw_pose(pose, h=None, w=None, canvas=None, draw_hands=True, draw_face=True):
    bodies = pose['bodies']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    if canvas is None:
        h = h or pose['H']
        w = w or pose['W']
        canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, candidate, subset)
    if draw_hands:
        canvas = draw_handpose(canvas, hands)
    if draw_face:
        faces = pose['faces']
        canvas = draw_facepose(canvas, faces)
    return canvas


if __name__ == "__main__":
    with open('./example.pkl', 'rb') as file:
        loaded_dict = pickle.load(file)

    print(loaded_dict)
    out = draw_pose(loaded_dict)
    plt.imsave('./1.jpg', out)
