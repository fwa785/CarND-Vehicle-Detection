import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import glob
import math
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
heat_threshold = 1
group_heat_frames = 5
vehicles = []
frame_index = [0]

class Vehicle(object):
    def __init__(self, position, size, frame_index):
        self.position = position
        self.size = size
        self.frame_count = 1
        self.last_found_frame_index = frame_index
        self.heat = 1
    def update_position(self, position, size, frame_index):
        distance = math.sqrt((self.position[0] - position[0])**2 +
                        (self.position[1] - position[1])**2)
        if ( distance < 50):
            self.position = position
            self.size = (int((self.size[0] + size[0])/2),
                         int((self.size[1] + size[1])/2))
            self.size = size
            self.frame_count += 1
            self.last_found_frame_index = frame_index
            return True
        else:
            return False

def find_vehicles_from_labels(labels, frame_index):

    boxes = find_boxes_from_labels(labels)
    frame_index[0] += 1

    for bbox in boxes:

        position = (int((bbox[0][0] + bbox[1][0])/2),
                    int((bbox[0][1] + bbox[1][1])/2))
        size = (abs(bbox[1][0] - bbox[0][0]),
                abs(bbox[1][1] - bbox[0][1]))

        vehicle_found = False
        for vehicle in vehicles:
            if (vehicle.update_position(position, size, frame_index[0])):
                vehicle_found = True
                if (vehicle.heat < 15):
                    vehicle.heat += 1
                break

        if (vehicle_found == False):
            vehicle = Vehicle(position, size, frame_index[0])
            vehicles.append(vehicle)

    for vehicle in vehicles:
        if (vehicle.last_found_frame_index < frame_index[0]):
            if (vehicle.heat > 0):
                vehicle.heat -= 1

def draw_vehicle_bboxes(img):
    for vehicle in vehicles:
        if ( vehicle.heat > 10):

            pt1 = (int(vehicle.position[0] - vehicle.size[0]/2),
                   int(vehicle.position[1] - vehicle.size[1]/2))
            pt2 = (int(vehicle.position[0] + vehicle.size[0]/2),
                   int(vehicle.position[1] + vehicle.size[1]/2))

            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 6)

    return img

def process_img(img):

    subsample_img1, heat1 = find_cars(img, 500, 650, 2, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)
    subsample_img, heat2 = find_cars(subsample_img1, 400, 500, 1, svc, X_scaler, orient, pix_per_cell,
                        cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)


    '''
    # apply threshold to heat
    heat1 = apply_threshold(heat1, heat_threshold)

    # apply threshold to heat
    heat2 = apply_threshold(heat2, heat_threshold)

    # combined the heat
    heat = heat1 + heat2
    '''

    heat = heat1 + heat2
    heat = apply_threshold(heat, heat_threshold)

    labels = label(heat)

    find_vehicles_from_labels(labels, frame_index)

    draw_img = draw_vehicle_bboxes(np.copy(img))
    #draw_img = draw_labeled_bboxes(np.copy(img), labels)


    return draw_img

video_filename = 'project_video.mp4'
video_output = 'output_videos/' + video_filename
clip = VideoFileClip(video_filename)
output_clip = clip.fl_image(process_img)
output_clip.write_videofile(video_output, audio=False)
