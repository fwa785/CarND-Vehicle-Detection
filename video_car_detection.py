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
cells_per_step = 1  # Instead of overlap, define how many cells to step
heat_threshold = 6
group_heat_threshold = 10
vehicles = []

class Vehicle(object):
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.frame_count = 1
        self.heat = 1
        self.found = True
    def update_position(self, position, size):
        distance = math.sqrt((self.position[0] - position[0])**2 +
                        (self.position[1] - position[1])**2)
        if ( distance < 50):
            self.position = position
            self.size = (int(self.size[0] * 0.8 + size[0] * 0.2),
                         int(self.size[1] * 0.8 + size[1] * 0.2))
            self.frame_count += 1
            self.found = True
            return True
        else:
            return False
    def increment_heat(self):
        if (self.heat < group_heat_threshold * 1.5):
            self.heat += 1
    def decrement_heat(self):
        if (self.heat > 0):
            self.heat -= 1

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

def draw_vehicle_bboxes(img):
    for vehicle in vehicles:
        if ( vehicle.heat >= group_heat_threshold):
            pt1 = (int(vehicle.position[0] - vehicle.size[0]/2),
                   int(vehicle.position[1] - vehicle.size[1]/2))
            pt2 = (int(vehicle.position[0] + vehicle.size[0]/2),
                   int(vehicle.position[1] + vehicle.size[1]/2))

            cv2.rectangle(img, pt1, pt2, (0, 0, 255), 6)

    return img

def process_img(img):

    heat = np.zeros_like(img[:, :, 0].astype(np.float))

    subsample_img  = img
    subsample_img, heat = find_cars(subsample_img, heat, 500, 650, 2, svc, X_scaler, orient, pix_per_cell,
                                      cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)
    subsample_img, heat = find_cars(subsample_img, heat, 400, 500, 1, svc, X_scaler, orient, pix_per_cell,
                                     cell_per_block, color_space, spatial_size, hist_bins, cells_per_step)

    heat = apply_threshold(heat, heat_threshold)

    labels = label(heat)

    find_vehicles_from_labels(labels)

    draw_img = draw_vehicle_bboxes(np.copy(img))

    return draw_img

video_filename = 'project_video.mp4'
video_output = 'output_videos/' + video_filename
clip = VideoFileClip(video_filename)
output_clip = clip.fl_image(process_img)
output_clip.write_videofile(video_output, audio=False)
