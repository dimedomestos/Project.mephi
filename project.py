import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
def process_video(path):
    cap = cv.VideoCapture(path)
    count = 0
    prev_frame = None
    model = YOLO("yolo11n-seg.pt")

    width = 1920
    height = 1080
    count_inc = 1200
    values_1 = []
    values_2 = []
    values_3 = []
    dilated_edges_prev = None
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv.resize(frame, (width, height))
            count += count_inc
            cap.set(cv.CAP_PROP_POS_FRAMES, count)
            #
            if prev_frame is None:
                prev_frame = frame
            results = model.predict(frame)

            masks = results[0].masks

            if masks is not None:
                mask = masks.data[0].numpy().astype(np.uint8)
                mask = cv.dilate(mask, kernel=np.ones((17, 17), dtype=np.uint8))
                mask = cv.resize(mask, (width, height))
                mask_inv = cv.bitwise_not(mask) // 255

                add = cv.bitwise_and(prev_frame, prev_frame, mask=mask)
                bg = cv.bitwise_and(frame, frame, mask=mask_inv)
                res = cv.bitwise_or(bg, add)
                prev_frame = res
            else:
                res = prev_frame = frame
            values_1.append(get_value(res))
            print(values_1[-1])
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(res, str(count//30), (30,60), font, 4, (255, 255, 255), 5, cv.LINE_AA)
            cv.imshow('Res', res)
            cv.waitKey(1)

            edges = cv.Canny(res, 100, 200, apertureSize=3)
            teacher_edges = cv.Canny(mask * 255, 100, 200, apertureSize=3)
            teacher_edges = cv.dilate(teacher_edges, kernel=np.ones((5, 5), dtype=np.uint8))
            t_edges_inv = cv.bitwise_not(teacher_edges)
            corrected_edges = cv.bitwise_and(edges, t_edges_inv)

            dilated_edges = cv.dilate(corrected_edges, kernel=np.ones((51, 51), dtype=np.uint8))
            values_2.append(dilated_edges.sum())
            if dilated_edges_prev is None:
                dilated_edges_prev = dilated_edges
            values_3.append(get_value_3(dilated_edges))
            dilated_edges_prev = dilated_edges
            cv.imshow('dilated', cv.resize(dilated_edges, (dilated_edges.shape[1] // 4, dilated_edges.shape[0] // 4)))

        else:
            print(count)
            cap.release()
            break
    t = [x*count_inc//30 for x in list(range(len(values_1)))]
    plt.plot(t, values_1)
    plt.show()
    plt.plot(t, values_2)
    plt.show()
    plt.plot(t, values_3)
    plt.show()
def get_value(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 125, 175)
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    return len(contours)

def get_clever_value(frame, prev_frame):
    x = 200
    y = (1080 * x) // 1980
    m = 1980 // (x * 4)
    thresh = 30
    matrix = cv.resize(frame, (x, y))
    prev_matrix = cv.resize(prev_frame, (x,y))

    missed = (matrix - prev_matrix) > thresh
    cv.imshow('missed', cv.resize(missed.astype(dtype = np.uint8) * 255, (x * m, y * m), interpolation=cv.INTER_NEAREST))
    cv.imshow('matrix', cv.resize(matrix.astype(dtype=np.uint8) * 255, (x * m, y * m), interpolation=cv.INTER_NEAREST))
    cv.imshow('prev', cv.resize(prev_matrix.astype(dtype=np.uint8) * 255, (x * m, y * m), interpolation=cv.INTER_NEAREST))
    cv.waitKey(500)

    return missed.sum()

def get_value_3(frame):
    x = 200
    y = (1080 * x) // 1980
    m = 1980 // (x * 4)
    matrix = cv.resize(frame, (x, y))
    cv.imshow('matrix', cv.resize(matrix.astype(dtype=np.uint8), (x * m , y*m), interpolation=cv.INTER_NEAREST))
    return matrix.sum()

#def max_point(


if name == "main":
    path = 'lecture2.mp4'

    process_video(path)
