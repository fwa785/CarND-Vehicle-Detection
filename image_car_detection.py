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


ystart = 350
ystop = 500
scale = 1
cells_per_step = 2  # Instead of overlap, define how many cells to step
heat_threshold = 1

images = glob.glob('test_images/*.jpg')

for file in images:
    img = mpimg.imread(file)

    t = time.time()
    subsample_img, heat1 = find_cars(img, 400, 650, 1.5, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)
    subsample_img, heat2 = find_cars(img, 300, 400, 1, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)

    heat = heat1 + heat2
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to find cars with hog sub sampling windows')

    # apply threshold to heat
    heat = apply_threshold(heat, heat_threshold)

    # Visualize the heapmap when display
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    print(labels[1], 'cars found')

    t = time.time()
    window_img, num_windows = window_search_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step, hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search', num_windows, "windows")

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(subsample_img)
    ax1.set_title("Hog Sub-Sampling Window Search")
    ax2.imshow(window_img)
    ax2.set_title("Naive Window Search")
    plt.show()

    plt.figure()
    plt.subplot(221)
    plt.imshow(subsample_img)
    plt.title("SubSampling Window Search")
    plt.subplot(222)
    plt.imshow(heatmap, cmap='hot')
    plt.title("Heap map")
    plt.subplot(223)
    plt.imshow(draw_img)
    plt.title("Label Boxes")
    plt.show()
