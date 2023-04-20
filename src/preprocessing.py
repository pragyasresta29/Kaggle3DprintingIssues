
import cv2
import os

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
basePath = "/Users/pragya/PycharmProjects/NLP/Kaggle3DprintingIssues/"

M_DIM = (1280, 720) # img aspect ratio of majority of images.
IMG_SIZE = 160 # Desired image size


def get_dim(width, height, resize):
    # computing new width & height to maintain aspect ratio
    r = resize / width
    dim = (resize, int(height * r))
    # print(dim)
    return dim


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    top_scale = img.shape[0] * 0.25

    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y = center_y - height_scaled / 2
    bottom_y = center_y + top_scale / 2
    # print(top_y, bottom_y, left_x, right_x)
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def show_img(img_arr):
    cv2.imshow("img", img_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_img(img, height):

    dim = get_dim(img.shape[1], img.shape[0], height)
    m_dim = get_dim(M_DIM[0], M_DIM[1], height)
    # The majority of the image have the following aspect ratio so we are going to maintain that while resizing
    # Cropping images with different aspect ratio
    if dim != m_dim:
        print("not equal")
        print(img.shape)
        img = crop_img(img)
        print(img.shape)
        dim = m_dim
    resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print(resize.shape)
    return resize


def convert_img_edges(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    return cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection


def prepare_data(df):
    data = []
    for index, row in df.iterrows():
        print(index)
        img_path = os.path.join(basePath + "data/images/", row['img_path'])
        try:
            img_arr = cv2.imread(img_path)[..., ::-1]  # convert BGR to RGB format
            resized_arr = resize_img(img_arr, IMG_SIZE)  # Reshaping images to preferred size
            data.append([resized_arr, int(row['has_under_extrusion'])])
        except Exception as e:
            print(e)
    return np.array(data, dtype=object)


def prepare_data_with_edges(df):
    data = []
    for index, row in df.iterrows():
        print(index)
        img_path = os.path.join(basePath + "data/images/", row['img_path'])
        try:
            img_arr = cv2.imread(img_path)[..., ::-1]  # convert BGR to RGB format
            img_arr = convert_img_edges(img_arr)  # get edges only
            resized_arr = resize_img(img_arr, IMG_SIZE)  # Reshaping images to preferred size
            data.append([resized_arr, int(row['has_under_extrusion'])])
        except Exception as e:
            print(e)
    return np.array(data, dtype=object)



# Fetch Train and Test Dataset
train_df = pd.read_csv(basePath + "data/train.csv")
test_df = pd.read_csv(basePath + "data/test.csv")

# Preprocessing Train Dataset
train_ds = prepare_data(train_df)

# Saving it into numpy array to use later
np.save(basePath + 'src/data/' + 'train_ds_160.npy', train_ds)


# Preprocess data with edge detection
train_ds_edge = prepare_data_with_edges(train_df)
np.save(basePath + 'src/data/' + 'train_ds_edges_80.npy', train_ds_edge)


def prepare_test_data(df):
    data = []
    for index, row in df.iterrows():
        print(index)
        img_path = os.path.join(basePath + "data/images/", row['img_path'])
        try:
            img_arr = cv2.imread(img_path)[..., ::-1]  # convert BGR to RGB format
            resized_arr = resize_img(img_arr, IMG_SIZE)  # Reshaping images to preferred size
            data.append([resized_arr, int(1)])
        except Exception as e:
            print(e)
    return np.array(data, dtype=object)


def prepare_test_data_with_edges(df):
    data = []
    for index, row in df.iterrows():
        print(index)
        img_path = os.path.join(basePath + "data/images/", row['img_path'])
        try:
            img_arr = cv2.imread(img_path)[..., ::-1]  # convert BGR to RGB format
            img_arr = convert_img_edges(img_arr)  # get edges only
            resized_arr = resize_img(img_arr, IMG_SIZE)  # Reshaping images to preferred size
            data.append([resized_arr, int(1)])
        except Exception as e:
            print(e)
    return np.array(data, dtype=object)


test_ds = prepare_test_data(test_df)
# Saving it into numpy array to use later
np.save(basePath + 'src/data/' + 'test_ds_160.npy', test_ds)

test_ds_edges = prepare_test_data_with_edges(test_df)
np.save(basePath + 'src/data/' + 'test_ds_edges_80.npy', test_ds_edges)






