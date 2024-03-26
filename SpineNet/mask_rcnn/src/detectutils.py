# Copyright 2021-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Used to analyze predictions."""
import csv
import logging
import warnings

import cv2
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')
warnings.filterwarnings("ignore")
COLOR_MAP = [
    (0, 165, 255),
    (0, 255, 0),
    (256, 0, 0),
    (0, 0, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 128, 128),
    (0, 128, 0),
    (128, 0, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
]


def write_list_to_csv(file_path, data_to_write, append=False):
    logging.info('Saving data into file [%s]...', file_path)
    if append:
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(file_path, open_mode) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_to_write)


def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False, None
    return True, image


def save_image(image_path, image):
    return cv2.imwrite(image_path, image)


def draw_rectangle(image, pt1, pt2, label=None):
    if label is not None:
        map_index = label % len(COLOR_MAP)
        color = COLOR_MAP[map_index]
    else:
        color = COLOR_MAP[0]
    thickness = 5
    cv2.rectangle(image, pt1, pt2, color, thickness)


def draw_text(image, text, org, label=None):
    if label is not None:
        map_index = label % len(COLOR_MAP)
        color = COLOR_MAP[map_index]
    else:
        color = COLOR_MAP[0]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    cv2.putText(image, text, org, font_face, font_scale, color, thickness)


def draw_one_box(image, label, box, cat_id, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    if cat_id is not None:
        map_index = cat_id % len(COLOR_MAP)
        color = COLOR_MAP[map_index]
    else:
        color = COLOR_MAP[0]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf // 2)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 6, [255, 255, 255],
                thickness=tf // 2, lineType=cv2.LINE_AA)


def draw_one_box_with_segm(image, label, box, cat_id, segm,
                           line_thickness=None):
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    if cat_id is not None:
        map_index = cat_id % len(COLOR_MAP)
        color = COLOR_MAP[map_index]
    else:
        color = COLOR_MAP[0]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf // 2)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 6, [255, 255, 255],
                thickness=tf // 2, lineType=cv2.LINE_AA)

    mask = segm != 0
    colored_mask = np.ones_like(image)
    colored_mask[mask] = color
    image[mask] = image[mask] // 2 + colored_mask[mask] // 2
