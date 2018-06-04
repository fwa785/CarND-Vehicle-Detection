import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

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


ystart = 400
ystop = 656
scale = 1.5
cells_per_step = 2  # Instead of overlap, define how many cells to step
heat_threshold = 12
group_heat_frames = 10

class Sum_Heat():
    def __init__(self, group_heat_frames):
        self.heat_history = []
        self.frame_index = 0
        self.group_heat_frames = group_heat_frames

    def add_heat(self, heat):
        if (len(self.heat_history) < self.group_heat_frames):
            self.heat_history.append(heat)
        else:
            self.heat_history[(self.frame_index % self.group_heat_frames)] = heat
        self.frame_index += 1

    def get_heat(self):
        sum_heat = np.zeros_like(self.heat_history[0])
        for index in range(0, len(self.heat_history)):
            sum_heat = np.add(sum_heat, self.heat_history[index])

        return sum_heat

sum_heat = Sum_Heat(group_heat_frames)

def process_img(img):
    subsample_img, heat1 = find_cars(img, 400, 650, 1.5, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)
    subsample_img, heat2 = find_cars(img, 300, 400, 1, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)

    heat = heat1 + heat2

    sum_heat.add_heat(heat)
    average_heat = sum_heat.get_heat()

    # apply threshold to heat
    average_heat = apply_threshold(average_heat, heat_threshold)

    # Find final boxes from heatmap
    labels = label(average_heat)

    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

video_filename = 'test_video.mp4'
video_output = 'output_videos/' + video_filename
clip = VideoFileClip(video_filename)
output_clip = clip.fl_image(process_img)
output_clip.write_videofile(video_output, audio=False)
