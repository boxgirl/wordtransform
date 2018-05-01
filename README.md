## Algorithm for localization and transformation of text for simplification the recongnizing

In the image, the text may be located at an angle, which may create some difficulty in recognizing.

The basis of the idea of ​​the method is that the decision is taken based on the results of processing one image through different filters. Input data to this program is a directory of images, containing separate words.

The use of these filters is due to the fact that each of them has its advantages in some non-standard situations, thus solving the problem of making a false decision as a result. Each image processes with the `Threshold ()` function, and then we get the coordinates of our pixels with a given threshold:

```filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x']```

That is, the required pixels will have a threshold of 0. For optimality, it was decided to choose exactly this threshold, because input images can be not only with black text, but also with white, and in order to avoid further problems you need to change the image to the desired state: white and black, and not the Threshold () standard black and white. 
```thresh =get_thresh(im)
thresh1= cv2.adaptiveThreshold
(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
coords = np.column_stack(np.where(thresh1 ==0))
```

After that, we turn to the calculation of the angles. In this block, the basic idea is to use for each filter the Fourier transformation, which means the image frequency matrix (the frequency field is obtained, which makes it easy to see the main lines of the image orientation). The resulting conversion is applied to each filter and we obtain new coordinates of the position of the problem text segments.

As a result, we have an array of angles of inclination for all filters, due to this we will calculate the statistics (sigma-statistics, to remove abnormal data in the sample).

After these transformations, a decision is made whether to rotate the image and after that we calculate the final matrix.

<p align="center"><img src="https://s9.postimg.cc/p99re3yb3/wordt.jpg" alt="Word transformations"/></p>

## How to install
```python setup.py install```
## How to use
-p - add your directory

```python3 wordtransform.py -p "PATH" -p "PATH"...```
