import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,cv2,keras
import constants
def get_iou2(bb1,bb2):
    assert bb1['x1']<bb1['x2']
    assert bb1['y1']<bb1['y2']
    assert bb2['x1']<bb2['x2']
    assert bb2['y1']<bb2['y2']
    #intersection area square coordinates
    x_left = max(bb1['x1'],bb2['x1'])
    x_right = min(bb1['x2'],bb2['x2'])
    y_bottom = min(bb1['y2'],bb2['y2'])
    y_top = max(bb1['y1'],bb2['y1'])

    if x_right < x_left or y_bottom<y_top:
        return 0.0
    intersection_area = (x_right - x_left)*(y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1'])*(bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1'])*(bb2['y2'] - bb2['y1'])
    iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
    assert iou >=0.0
    assert iou <=1.0
    return iou

bb1 = {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}
bb2 = {'x1': 30, 'y1': 30, 'x2': 70, 'y2': 70}

# Compute IoU
iou = get_iou2(bb1, bb2)
print("IoU:", iou)

