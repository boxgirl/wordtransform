#!/usr/bin/env python3
import timeit
import math
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import click

@click.command()
@click.option('--path', '-p', help='Path of directory with data', multiple=True)
def prep_for_proc(path):
    for p in path:
        for filename in glob.glob(p):
            image_list.append(filename)
        image_processing()

image_list = []
def get_mean_filter():
    return  np.ones((3, 3))
def get_gauss():
    x = cv2.getGaussianKernel(5, 10)
    return  x * x.T
# different edge detecting filters
def get_scharr():
    return np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]])
def get_sobel_x():
    return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
def get_sobel_y():
    return np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
def get_laplacian():
    return np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

def get_thresh(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    return  cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]

def get_mag_spect(thresh):
    f = np.fft.fft2(thresh)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift))

def get_rotated_variants(angle1,rotated,coords,total_angle,im):
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    total_angle = total_angle + angle
    angle1.append(angle)
    (h, w) = im.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated.append(cv2.warpAffine(im, M, (w, h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    return total_angle

def get_fin_angle(angle1,total_angle):
    sqrt_ang=0
    fin_angle = total_angle / angle1.__len__()
    for k in angle1:
        sqrt_ang += (k - fin_angle) ** 2
    sigma = np.sqrt(sqrt_ang / angle1.__len__())
    total_angle = 0
    angle=list()
    for a in angle1:
        if (math.fabs(a) <= math.fabs(2*sigma)): angle.append(a)
    for f in angle:
        total_angle += f
    return total_angle / angle.__len__()

def get_plot(im,rotated,fin_angle):
    print('[INFO] angle: {:.3f}'.format(fin_angle))
    filter_name = ['Input', 'minAreaRect', 'mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x',
                   'RESULT','persp']
    images = [im, rotated[0], rotated[1], rotated[2], rotated[3], rotated[4], rotated[5], rotated[6], rotated[7],rotated[8]]
    for i in range(images.__len__()):
        plt.subplot(2, 5, i + 1), plt.imshow(images[i], cmap='gray')
        plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
    plt.show()

def image_processing():
    filters = [get_mean_filter(), get_gauss(), get_laplacian(), get_sobel_x(), get_sobel_y(), get_scharr()]
    for i in  image_list:
        a = timeit.default_timer()
        rotated=list()
        angle1=list()
        im=cv2.imread(i)
        im= cv2.resize(im, (200, 200))
        thresh =get_thresh(im)
        thresh1= cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        coords = np.column_stack(np.where(thresh1 ==0))
        #rect = cv2.minAreaRect(coords)
        #box = cv2.cv.BoxPoints(rect)
        #box = np.int0(box)
        total_angle = 0.0
        #get minAreaMatrix
        total_angle=get_rotated_variants(angle1,rotated,coords,total_angle,im)
        magnitude_spectrum = get_mag_spect(thresh)
        for lol in filters:
            coords = np.column_stack(np.where( (cv2.filter2D(magnitude_spectrum,-1,lol))>175))
           #get filtersMatrix
            total_angle = get_rotated_variants(angle1,rotated,coords, total_angle, im)

        fin_angle=get_fin_angle(angle1,total_angle)
        #get resultMatrix
        (h, w) = im.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, fin_angle, 1.0)
        rotated.append(cv2.warpAffine(im, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))

        pts1 = np.float32([[0, 0], [h, 0], [0,w], [h, w]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(rotated[rotated.__len__()-1], M, (300, 300))
        rotated.append(dst)

        print(timeit.default_timer() - a)
        get_plot(im,rotated,fin_angle)

if __name__ == '__main__':
    prep_for_proc()
