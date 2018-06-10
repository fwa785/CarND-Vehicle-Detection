import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import glob
from scipy.ndimage.measurements import label

from car_detection import *

# load a pre-trained svc model from a serialized file
svc_pickle = pickle.load(open("svc_pickle.p", "rb"))

svc = svc_pickle["svc"]
X_scaler = svc_pickle["scaler"]
color_space = svc_pickle["color_space"]
orient = svc_pickle["orient"]
pix_per_cell = svc_pickle["pix_per_cell"]
cell_per_block = svc_pickle["cell_per_block"]
spatial_size = svc_pickle["spatial_size"]
hist_bins = svc_pickle["hist_bins"]
hog_channel = svc_pickle["hog_channel"]
spatial_feat = svc_pickle["spatial_feat"]
hist_feat = svc_pickle["hist_feat"]
hog_feat = svc_pickle["hog_feat"]


cells_per_step = 1 # Instead of overlap, define how many cells to step
heat_threshold = 4

images = glob.glob('test_images/*.jpg')

for file in images:
    img = mpimg.imread(file)

    ystart = 350
    ystop = 650
    scale = 2

    # First display the search window for different overlaps
    overlap_8_img, num_windows = window_search_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step * 8, hog_channel,
                        draw_all_windows=True)

    overlap_4_img, num_windows = window_search_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step * 4, hog_channel,
                        draw_all_windows = True)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(overlap_8_img)
    ax1.set_title("(128, 128) 8 cells overlap")
    ax2.imshow(overlap_4_img)
    ax2.set_title("(128, 128) 4 cells overlap")
    plt.show()

    # Compare the time for two algorithms

    # First get the time for subsampling algorithm
    t = time.time()
    heat = np.zeros_like(img[:, :, 0].astype(np.float))
    subsample_img, heat = find_cars(img, heat, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                    cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to find cars with hog sub sampling windows')

    # Second, use the algorithm to extract HOG feature for each sliding window
    t = time.time()
    window_img, num_windows = window_search_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step, hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search', num_windows, "windows")

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(subsample_img)
    ax1.set_title("Hog Sub-Sampling Window Search")
    ax2.imshow(window_img)
    ax2.set_title("Hog Extract for Sliding Windows")
    plt.show()

    # Show the search Regions
    # The first region of interest using scale=2
    ystart = 350
    ystop = 650
    scale = 2

    search_region_img = np.copy(img)
    search_region_img, heat = find_cars(search_region_img, heat, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                      cell_per_block, color_space, spatial_size, hist_bins, cells_per_step,
                                    show_all_windows=True)
    # The second region of interest using scale=1
    ystart = 400
    ystop = 500
    scale = 1
    search_region_img, heat = find_cars(search_region_img, heat, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                     cell_per_block, color_space, spatial_size, hist_bins, cells_per_step,
                                    box_color=(255, 0, 0), show_all_windows=True)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax2.imshow(search_region_img)
    ax2.set_title("Search Regions")
    plt.show()

    # Now let's use subsampling algorithm to search for the cars
    heat = np.zeros_like(img[:, :, 0].astype(np.float))

    # The first region of interest using scale=2
    ystart = 350
    ystop = 650
    scale = 2
    subsample_img, heat = find_cars(img, heat, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                      cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)


    # The first region of interest using scale=1
    ystart = 400
    ystop = 500
    scale = 1
    subsample_img, heat = find_cars(img, heat, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                     cell_per_block, color_space, spatial_size, hist_bins, cells_per_step,
                                    box_color=(255, 0, 0))

    # apply threshold to heat
    heat = apply_threshold(heat, heat_threshold)

    # Visualize the heapmap when display
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    print(labels[1], 'cars found')

    # Plot the heatmap
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax2.imshow(heatmap, cmap='hot')
    ax2.set_title("Heatmap")
    ax3.imshow(draw_img)
    ax3.set_title("Cars Found")
    plt.show()

