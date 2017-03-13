##Vehicle Detection 

This project detects vehicles in a given video stream of highway data. There are multiple ways to do this; this project focuses on an approach with HOG feature extraction + SVM classifier training. The code for this project is contained in the `code.py` file. 

## Inspect Data

The code starts off with a quick visual inspection of the data 

![Image](https://github.com/kiranganesh/CarND-Vehicle-Detection/blob/master/examples/image1.JPG)

## HOG Features

The HOG features are extracted with the help of the `get_hog_features` function provided in the Udacity lessons. Using some test code to sweep through the various HOG extraction parameeters (orient, pix_per_cell and cells_per_block), I looked at a few different values of the parameters.

Some sample data is given below for both vehicle and non-vehicle data. The title (x,y,z) of each picture shows the values of (orient, pix_per_cell and cells_per_block)

![Image](https://github.com/kiranganesh/CarND-Vehicle-Detection/blob/master/examples/image2.JPG)

![Image](https://github.com/kiranganesh/CarND-Vehicle-Detection/blob/master/examples/image3.JPG)

I also explored different color spaces and ended up picking YUV color space with orientation of 11, pix_per_cell of 16 and cell_per_block of 2. After multiple trial runs, there were also many other combinations that produced somewhat similar result so I'm not sure that the values I picked were necessarily the most optimal. 

For initial build of video extraction pipeline, I decided to use only the HOG feature extraction. The color histogram/spatial color info were additional tools I could come back to and rely on if the HOG features did not provide sufficient accuracy by themselves.

## SVC Classifier

The SVC was extremely simple to create and train using the extracted feature data, requiring essentially only 7 lines of code:

![Image](https://github.com/kiranganesh/CarND-Vehicle-Detection/blob/master/examples/image4.JPG)

The SVC was able to achieve a test data accuracy of 0.9837 for classification. 

## Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

