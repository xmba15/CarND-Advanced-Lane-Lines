#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
from image_processing import preprocess_single_image


class LaneLineDetector(object):
    def __init__(self, img, mtx=None, dist=None, offset=200, record_process=False, tolerance=1e-6):
        assert img.ndim == 3

        self.__record_process = record_process
        self.__tolerance = tolerance

        if self.__record_process:
            print("img of type BGR")

        self.img = img
        if mtx is not None and dist is not None:
            self.img = cv2.undistort(self.img, mtx, dist, None, mtx)

        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.HEIGHT = self.img.shape[0]
        self.WIDTH = self.img.shape[1]
        self.offset = offset

        self.__src = src = np.float32(
            [
                [self.WIDTH // 2 - 60, 450],
                [self.WIDTH // 2 + 60, 450],
                [100, self.HEIGHT],
                [self.WIDTH - 100, self.HEIGHT],
            ]
        )

        self.__dst = dst = np.float32(
            [
                [self.offset, self.offset],
                [self.WIDTH - self.offset, self.offset],
                [self.offset, self.HEIGHT],
                [self.WIDTH - self.offset, self.HEIGHT],
            ]
        )

        self.xm_per_pix = 3.7 / (self.WIDTH - 200)
        self.ym_per_pix = 30 / 720

        self.M = cv2.getPerspectiveTransform(self.__src, self.__dst)
        self.Minv = cv2.getPerspectiveTransform(self.__dst, self.__src)

    def transform_perspective(self, img, M):
        assert img.ndim == 3
        HEIGHT, WIDTH = img.shape[:2]

        return cv2.warpPerspective(img, M, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    def thresholding_process(self, img):
        assert img.ndim == 3
        return preprocess_single_image(img)

    def estimate_histogram(self, img):
        assert img.ndim == 2
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return histogram

    def find_lane_pixels(self, binary_warped, nwindows=9, margin=100, minpix=50):
        assert binary_warped.ndim == 2

        HEIGHT, WIDTH = binary_warped.shape[:2]
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        histogram = self.estimate_histogram(binary_warped)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(HEIGHT // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = HEIGHT - (window + 1) * window_height
            win_y_high = HEIGHT - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if self.__record_process:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, HEIGHT, WIDTH, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, HEIGHT - 1, HEIGHT)

        try:
            left_fitx = left_fit[0] * ploty ** 2 + \
                left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + \
                right_fit[1] * ploty + right_fit[2]
        except TypeError:
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def fit_polynomial_without_prior(self, binary_warped):
        assert binary_warped.ndim == 2
        HEIGHT, WIDTH = binary_warped.shape[:2]

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(
            binary_warped)

        left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_poly(
            HEIGHT, WIDTH, leftx, lefty, rightx, righty)

        if self.__record_process:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img

    def fit_polynomial_with_prior(self, binary_warped, left_fit, right_fit, margin=100):
        assert binary_warped.ndim == 2
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            nonzerox > (left_fit[0] * (nonzeroy ** 2) +
                        left_fit[1] * nonzeroy + left_fit[2] - margin)
        ) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
        right_lane_inds = (
            nonzerox > (right_fit[0] * (nonzeroy ** 2) +
                        right_fit[1] * nonzeroy + right_fit[2] - margin)
        ) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        new_left_fit, new_right_fit, left_fitx, right_fitx, ploty = self.fit_poly(
            binary_warped.shape[0], binary_warped.shape[1], leftx, lefty, rightx, righty
        )

        if self.__record_process:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

        return (new_left_fit, new_right_fit, left_fitx, right_fitx, ploty, out_img)

    def estimate_lanes(self, left_fit=None, right_fit=None):
        transformed_img = self.transform_perspective(self.img, self.M)
        thresholded_img = self.thresholding_process(transformed_img)
        if left_fit is not None and right_fit is not None:
            left_fit_param, right_fit_param, left_fitx, right_fitx, ploty, out_img = self.fit_polynomial_with_prior(
                thresholded_img, left_fit, right_fit
            )
        else:
            left_fit_param, right_fit_param, left_fitx, right_fitx, ploty, out_img = self.fit_polynomial_without_prior(
                thresholded_img
            )

        if self.__record_process:
            self.transformed_img = transformed_img
            self.thresholded_img = thresholded_img
            for x, y in zip(left_fitx, ploty):
                if x >= self.WIDTH or y >= self.HEIGHT:
                    continue
                out_img[int(y), int(x)] = [255, 185, 15]
            for x, y in zip(right_fitx, ploty):
                if x >= self.WIDTH or y >= self.HEIGHT:
                    continue
                out_img[int(y), int(x)] = [255, 185, 15]
            self.out_img = out_img

        return (thresholded_img, left_fit_param, right_fit_param, left_fitx, right_fitx, ploty, out_img)

    def estimate_lane_parameters(self, left_fit, right_fit, left_fitx, right_fitx, ploty):
        ego_car_position = self.WIDTH / 2
        left_fit_cr = np.polyfit(
            ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)
        y_eval = np.max(ploty)
        left_curvature_rad = (
            (1 + (2 * left_fit_cr[0] * y_eval *
                  self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * left_fit_cr[0])

        right_curvature_rad = (
            (1 + (2 * right_fit_cr[0] * y_eval *
                  self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])

        left_lane_bottom = (left_fit[0] * y_eval) ** 2 + \
            left_fit[0] * y_eval + left_fit[2]
        right_lane_bottom = (
            right_fit[0] * y_eval) ** 2 + right_fit[0] * y_eval + right_fit[2]

        ego_car_position = (
            self.WIDTH / 2 - (left_lane_bottom + right_lane_bottom) / 2) * self.xm_per_pix

        return (left_curvature_rad + right_curvature_rad) / 2, ego_car_position

    def visualize_result(self, thresholded_img, left_fit, right_fit, left_fitx, right_fitx, ploty):
        warp_zero = np.zeros_like(thresholded_img).astype(np.uint8)

        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32(
            [pts_left]), isClosed=False, color=(255, 0, 255), thickness=15)
        cv2.polylines(color_warp, np.int32(
            [pts_right]), isClosed=False, color=(0, 255, 255), thickness=15)

        newwarp = cv2.warpPerspective(
            color_warp, self.Minv, (self.WIDTH, self.HEIGHT))
        result = cv2.addWeighted(self.img, 1, newwarp, 0.5, 0)

        road_curvature_rad, ego_car_position = self.estimate_lane_parameters(
            left_fit, right_fit, left_fitx, right_fitx, ploty
        )

        cv2.putText(
            result, "Radius of Curvature = {} (m)".format(
                int(road_curvature_rad)), (100, 100), 2, 1, (255, 255, 0), 2
        )

        if ego_car_position < 0:
            position_text = "Vehicle is {:2f}m left of center".format(
                np.abs(ego_car_position))
        else:
            position_text = "Vehicle is {:2f} right of center".format(
                ego_car_position)

        cv2.putText(result, position_text, (100, 150), 2, 1, (255, 255, 0), 2)

        return result
