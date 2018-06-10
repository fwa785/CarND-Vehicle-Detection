# Vehicle Detection Project
## Overview

---

This project uses SVM model to train on the HOG features for car and not car images, then use the trained classifier to 
search for the vehicles in images and videos. The project includes the following steps:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Use SVM to train on the feature vectors being extracted for car and not car images 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_not_car_hog.png
[image3]: ./examples/window_search_overlap.png
[image4]: ./examples/sliding_window_comparison.png
[image5]: ./examples/window_search_regions.png
[image6]: ./examples/bboxes_and_heat_test1.png
[image7]: ./examples/bboxes_and_heat_test2.png
[image8]: ./examples/bboxes_and_heat_test3.png
[image9]: ./examples/bboxes_and_heat_test4.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

My code is at [here](https://github.com/fwa785/CarND-Vehicle-Detection).

---

## Car Not Car Classifier Training

The classifier training code is in train_feature.py file. It does the following steps:
* Read the Car, Not Car sample images
* Extract the features from the sample images
* Randomly split the sample features into 80% training set and 20% test set
* Train the classifier based on the features from the training set
* Verify the accuracy of the classifier on the test set
* Adjust parameters based on the accuracy on the test set

### Car NotCar Images
I started by reading in all the `vehicle` and `non-vehicle` images. These example images come from a combination of 
the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.
There are total 8792 car images and 8968 not car images.

Number of car images: 8792

Number of notcar images: 8968

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

### Feature Exaction
The extract_features() function in helper_function.py file extracts HOG feature 
for one color channel or all color channels, plus the spatial 
color spatially binned color and histograms of color in the feature vector.

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images

Function to extract HOG feature for one channel is provided in helper_functions.py file, and it's named get_hog_features(). 
The function takes one channel data of the image, and different parameters for hog feature extraction.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` 
and `cells_per_block=2`:

![alt text][image2]

#### 2. Choose HOG parameters

I tried different color spaces and the different HOG parameters, and use the training accuracy to evaluate which choices
are the best. 

Honestly, I didn't noticed too much difference for the training accuracy with different color 
spaces except RGB color space did give worse training accuracy. Not sure it is because that I choose ALL CHANNEL for
hog channels, so the difference is not obvious. 

I also tried different setting for parameters: orientations, pixels_per_cell, and cells_per_block. 
The best result I could get is to set `orientation=9`, `pixels_per_cell=8` and `cells_per_block=2`.

#### 3. Extract spatial color feature
The code to extract spatial color feature is in helper_functions.py, and the function name is bin_spatial(). 
Spatial color feature is the raw color feature of the image. The image is resized to smaller size, but the
raw color still retain features of the image to classify the image.

#### 4. Extract color histogram feature
The color histogram feature is extracted by function color_hist() in helper_functions.py file. The image is
converted to the selected color space before doing the color histogram feature exaction. This function finds
the histogram feature for each color channel, and combined the features from different color channels into one.

#### 5. Combine the Features
The features for a single image is combined by extract_single_image_features() function in helper_function.py.

Function extract_features() in helper_functions.py does the feature extraction for a list of images.

### Train Classifier

The features are randomly divided into 80% training data and 20% test data. Then the features are normalized. The code
is in train_feature.py.

Then I use linear SVM to train the classifier on the training features. The code to train the classifier 
is in train_feature.py.

Finally I use the test data to verify the accuray of the classifier. I could get about 97% of the accuracy.

The trained classifier's information and parameters are saved into svc_pickle.p file, and this is done in 
train_feature.py.

## Sliding Window Search

### 1. Sliding Window Search Algorithm  

Regions of interest are selected in the image, and window slides from left to right, up to down in the regions
of interest to check which window is classified as car. The sliding windows can be overlapped, and an overlapped
size is defined. The picture below shows the window search with different overlapping.

![alt text][image3]

I tried two methods of sliding window search. The first one extracts the HOG features for each sliding window. 
The second method extracts the HOG features for the entire image once, and use the subsample window to get the 
HOG feature for each sliding window. For both methods, the extracted features are classified by the trained 
classifier to identify whether the image in the window is car or not. Both methods are implemented in car_detection.py.

Both methods use the start and end point on y to select the region of interest for the car detection, they both take 
scale parameter to define the sliding window size based on (64, 64). If scale is 1, the sliding window is (64, 64), 
if scale is 2, the sliding window is (128, 128). Both methods use cells_per_step to define the overlap of the sliding 
window. Because extracting HOG features take long time, the second method is more efficient. For example, set to the 
same region of interest, and the same search window size, the time to extract the features for using hog sub sampling 
and sliding window search is shown below. As you can see the method to extract the feature on each sliding window takes
almost 8 times longer than the subsampling method.

```python
1.16 Seconds to find cars with hog sub sampling windows
8.72 Seconds to search 803 windows
```

The figure below shows the car being find by the two methods, the results are pretty similar

![alt text][image4]

Because the HOG subsampling window method is more efficient, so I use the subsampling window method for the car detection 
on the video clip.

### 2. Search Region and Scales
After some experiements, two regions of interest are selected for the car detection. One with Y starts from 350 to 650, with
search window size (128, 128). The second one with Y starts from 400 to 500, with search window size (64, 64). Window (128, 128)
is mainly for finding cars more close, and window (64, 64) is for finding cars further.

The figure below shows the regions of interest. Because the sliding overlap is set to 1 cell per step, both scales shows every dense
search grids. But they actually are using different sizes of sliding windows.

![alt text][image5]

### 3. Test Pipeline On The Test Images
Ultimately I choose using YCrCb all channel HOG features plus spatial color spatially binned color and 
histograms of color in the feature vector. Then choose two regions and two scales to search on as mentioned above. 

A heat matrix is used to filter out false detection. If the classifier detects the car in one sliding window, the heat
in the sliding window increment. The heat searched by each region of interest with different sliding window scale also 
adds up. Eventually, a threshold for heat is selected to filter out the false positive only detected by very few sliding 
windows. I selected the heat threshold to 4.

With the above setting, I got pretty good result. The code is in image_car_detection.py file.

Below are some results of some test images:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

---

## Video Implementation

### 1. Pipeline on Video
The pipeline is applied to the project video file. The implementation to do car detection on video is in file 
video_car_detection.py. It chooses the same parameters as the image car detection. The cars are detected pretty
well through most of the video. Occasionally the cars on the opposite lane are detected, but those are cars, so
I assume it is OK.

Here's a [link to my video result](./output_videos/project_video.mp4)


### 2. Remove False Positive

First some of the false positives are already filtered out by the heatmap implemented per image. 
However, I still got lots of false positives. So I used additional technicals to filter out false
positives. 

First, I track the position the each found vehicles. In the next frame, if a vehicle's position is
less than 30 from a vehicle in the previous frame, they're treated as the same vehicle. Next, when a vehicle
is found in a frame, it's heat incremented by 1, when a vehicle is not found in a frame, its heat decrements
by 1. This feature is implemented in the find_vehicles_from_label function in video_car_detection.py file as shown
below.

```python
def find_vehicles_from_labels(labels):

    boxes = find_boxes_from_labels(labels)

    # First set all vehicles to not found
    for vehicle in vehicles:
        vehicle.found = False

    # Search the vehicle matching the boxes being found
    for bbox in boxes:

        position = (int((bbox[0][0] + bbox[1][0])/2),
                    int((bbox[0][1] + bbox[1][1])/2))
        size = (abs(bbox[1][0] - bbox[0][0]),
                abs(bbox[1][1] - bbox[0][1]))

        vehicle_found = False
        for vehicle in vehicles:
            if (vehicle.update_position(position, size)):
                vehicle_found = True
                vehicle.increment_heat()
                break

        if (vehicle_found == False):
            vehicle = Vehicle(position, size)
            vehicles.append(vehicle)

    # decrement the heat for vehicle if it's not found for this frame
    for vehicle in vehicles:
        if (vehicle.found == False):
            vehicle.decrement_heat()
```

Eventually, only draw the box if the accumulated heat of the vehicle is above some threshold as shown below.

```python
def draw_vehicle_bboxes(img):
    for vehicle in vehicles:
        if ( vehicle.heat >= group_heat_threshold):
            pt1 = (int(vehicle.position[0] - vehicle.size[0]/2),
                   int(vehicle.position[1] - vehicle.size[1]/2))
            pt2 = (int(vehicle.position[0] + vehicle.size[0]/2),
                   int(vehicle.position[1] + vehicle.size[1]/2))

            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 6)

    return img

```

With this technicals, the false positives are removed from the video, and the result is pretty good.    

---

## Discussion

### Problems

First, my implementation is still not very good at detection vehicles at the edge or right corner. I think that could
be because I didn't try hard to improve my classifier. The classifier use the default sample images provided by the 
course, and I use Linear SVM to train the classifier. If I use more sample images, or use CNN to train the classifier, 
it should get a better result for identifying the vehicles in the corner. 

Second, very occasionally there are double boxes on one vehicle. I think that is because when the vehicle moves fast, the
positions between the frame is greater than 30, and I treated them as two vehicles. If I relax the threshold, it will 
have more false detection. This may be fine tuned. 
 
Last, the accumulated heat over frame algorithm I implmented may not be an efficient way to filter out the false positive.
It is probably better to simply average out the heat matrix over the last N frames, and then use the heat threshold to 
filter out the false positive on the averaged heat matrix. 

