import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

def convert_color(img, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        return np.copy(img)

def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)

    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thickness=thick)

    return draw_img

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):

    if vis == True:
        feature, hog_image =  hog(img, orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),
                                   cells_per_block=(cell_per_block,cell_per_block),visualise=vis,
                                   feature_vector=feature_vec,block_norm="L2-Hys")
        return feature, hog_image
    else:
        feature = hog(img, orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),
                                   cells_per_block=(cell_per_block,cell_per_block),visualise=vis,
                                   feature_vector=feature_vec,block_norm="L2-Hys")
        return feature

# define function to computer color histogram features
def bin_spatial(img, size=(32,32)):
    # first resize the image
    resized = cv2.resize(img, size)

    features = resized.ravel()
    return features

# define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], nbins, bins_range)
    channel2_hist = np.histogram(img[:,:,1], nbins, bins_range)
    channel3_hist = np.histogram(img[:,:,2], nbins, bins_range)
    # generate bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:-1])/2
    # concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0],
                                    channel3_hist[0]))

    return hist_features

def extract_single_image_features(img, color_space='RGB', spatial_size=(32,32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []

    feature_image = convert_color(img, color_space)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features.append(hist_features)
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            channel = hog_channel
            hog_features = get_hog_features(feature_image[:, :, channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True)
        features.append(hog_features)

    return np.concatenate(features)

def extract_features(imgs, color_space='RGB', spatial_size=(32,32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        image = mpimg.imread(file)

        file_features = extract_single_image_features(image, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(file_features)

    return features

def sliding_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                   xy_window=(64,64), xy_overlap=(0.5,0.5)):
    if x_start_stop[0] == None:
        x_start = 0
    else:
        x_start = x_start_stop[0]

    if x_start_stop[1] == None:
        x_end = img.shape[1]
    else:
        x_end = x_start_stop[1]

    if y_start_stop[0] == None:
        y_start = 0
    else:
        y_start = y_start_stop[0]

    if y_start_stop[1] == None:
        y_end = img.shape[0]
    else:
        y_end = y_start_stop[1]

    window_list = []

    x = x_start
    x_step = int(xy_window[0] * (1-xy_overlap[0]))
    while (x + xy_window[0] <= x_end):
        y = y_start
        y_step = int(xy_window[1] * (1-xy_overlap[1]))
        while (y + xy_window[1] <= y_end):
            window_list.append(((x, y),
                                (x + xy_window[0], y + xy_window[1])))

            y = y + y_step
        x = x + x_step

    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32,32), hist_bins=32,
                   hist_range=(0,256), orient=9,
                   pix_per_cell = 8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True, hist_feat=True,
                   hog_feat=True):

    # create a list to receive positive detection windows
    on_windows = []

    # Iterate over all windows
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]: window[1][0]], (64,64))
        # Extract features for that window
        features = extract_single_image_features(test_img,color_space=color_space,spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        # scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # predict using your classifier
        prediction = clf.predict(test_features)
        # If positive then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive deetection
    return on_windows

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def find_boxes_from_labels(labels):
    boxes = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)

    return boxes

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img
