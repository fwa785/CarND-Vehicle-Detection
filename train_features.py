import glob
import time
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helper_functions import *

# load the image name for cars
basedir = 'vehicles/'
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir + imtype + '/*'))

# load the image name for not cars
basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir + imtype + '/*'))

print('Number of car images:', len(cars))
print('Number of notcar images:', len(notcars))

# random shuffle the cars and notcars array
np.random.shuffle(cars)
np.random.shuffle(notcars)

# Reduce the sample size for experimenting the training hyper parameters
#sample_size = 2000
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

#display a car and not car image
car_index = np.random.randint(0, len(cars))
car_image = mpimg.imread(cars[car_index])
notcar_index = np.random.randint(0, len(notcars))
notcar_image = mpimg.imread(notcars[notcar_index])

# display car and not car image
f, (ax1, ax2)= plt.subplots(1, 2, figsize=(4, 2))
ax1.imshow(car_image)
ax1.set_title('Car')
ax2.imshow(notcar_image)
ax2.set_title('Not Car')
plt.show()

# hyper parameters
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

# get channel 1 hog features for a single image
car_feature, car_hog_img = get_hog_features(convert_color(car_image, color_space)[:, :, 2], orient,
                                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)

notcar_feature, notcar_hog_img = get_hog_features(convert_color(notcar_image, color_space)[:, :, 2], orient,
                                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)

# display car and not car hog image
f, (ax1, ax2, ax3, ax4)= plt.subplots(1, 4, figsize=(8, 2))
ax1.imshow(car_image)
ax1.set_title('Car')
ax2.imshow(car_hog_img, cmap='gray')
ax2.set_title('Car Hog Ch1')
ax3.imshow(notcar_image)
ax3.set_title('Not Car')
ax4.imshow(notcar_hog_img, cmap='gray')
ax4.set_title('Not Car Hog Ch1')
plt.show()

# Get features for Cars
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,hog_channel=hog_channel,
                                spatial_feat=spatial_feat,hist_feat=hist_feat,hog_feat=hog_feat)

# Get features for not cars
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,hog_channel=hog_channel,
                                spatial_feat=spatial_feat,hist_feat=hist_feat,hog_feat=hog_feat)

#Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Define label vectors
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split the data into training and test
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fot a per-column Scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Feature vector length:', len(X_train[0]))

# Use linear support vector
svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

#Check the score for SVC
print('Using:', color_space, "color space,", orient, "orientations",
      pix_per_cell, 'pixels per cell', cell_per_block, 'cells per block',
      hog_channel, 'hog channel', spatial_size, "spatial size",
      hist_bins, "histogram bins")
print('Test Accuracy for SVC =', round(svc.score(X_test, y_test), 4))

# Save the result to a pickle file
svc_filehandler = open('svc_pickle.p', 'wb')
# set attributes of SVC object
svc_pickle = {}
svc_pickle["svc"] = svc
svc_pickle["scaler"] = X_scaler
svc_pickle["color_space"] = color_space
svc_pickle["orient"] = orient
svc_pickle["pix_per_cell"] = pix_per_cell
svc_pickle["cell_per_block"] = cell_per_block
svc_pickle["spatial_size"] = spatial_size
svc_pickle["hist_bins"] = hist_bins
svc_pickle["hog_channel"] = hog_channel
svc_pickle["spatial_feat"] = spatial_feat
svc_pickle["hist_feat"] = hist_feat
svc_pickle["hog_feat"] = hog_feat
pickle.dump(svc_pickle, svc_filehandler)
