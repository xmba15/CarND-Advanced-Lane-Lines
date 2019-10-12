#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def threshold_single_channel_img(img, thresh_min, thresh_max):
    assert(img.ndim == 2)

    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh_min)
                  & (img <= thresh_max)] = 1

    return binary_output


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255,
                     kernel_size=3):
    assert(orient in ['x', 'y'])

    gray = np.copy(img)
    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel_single_axis = cv2.Sobel(gray, cv2.CV_64F, 1, 0, kernel_size)
    else:
        sobel_single_axis = cv2.Sobel(gray, cv2.CV_64F, 0, 1, kernel_size)

    abs_sobelx = np.absolute(sobel_single_axis)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    return threshold_single_channel_img(scaled_sobel, thresh_min=thresh_min,
                                        thresh_max=thresh_max)


def sobel_gradient_direction(img):

    gray = np.copy(img)
    if (img.ndim == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradient = np.hypot(sobelx, sobely)
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(sobelx, sobely)

    return (gradient, theta)


def hls(img):
    assert(img.ndim == 3)

    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return hls_image


def hsv(img):
    assert(img.ndim == 3)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_image


def thresholding_on_color(img):
    hsv_img = hsv(img)
    thresholding_on_yellow = cv2.inRange(
        hsv_img, (15, 100, 100), (30, 255, 255))
    thresholding_on_white = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    combined_result = np.zeros_like(hsv_img[:, :, 0])
    combined_result[(thresholding_on_yellow > 0) |
                    (thresholding_on_white > 0)] = 1
    return combined_result


def preprocess_single_image(img):
    assert(img.ndim == 3)
    hls_image = hls(img)
    s_channel = hls_image[:, :, 2]
    thresholded_on_color = thresholding_on_color(img)

    thresholded_sobel_x = abs_sobel_thresh(
        s_channel, orient='x', thresh_min=15, thresh_max=100)

    thresholded_binary_image = np.zeros_like(s_channel)
    thresholded_binary_image[(thresholded_sobel_x == 1)
                             | (thresholded_on_color == 1)] = 1

    return thresholded_binary_image
