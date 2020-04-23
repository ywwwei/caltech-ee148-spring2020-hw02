import json
import os

import numpy as np
from PIL import Image
import random


def normalize(patch):
    '''
    This function takes an numpy array <patch> and returns the normalizedn result
    :param patch:
    :return:
    '''
    if np.linalg.norm(patch):
        return patch / np.linalg.norm(patch)
    else:  # zeros
        return patch


def compute_convolution(I, T, stride=(1, 1), paddding=(0, 0), window_size=None, weight=np.array([1 / 3, 1 / 3, 1 / 3])):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality.
    :param I:
    :param T:
    :param stride:
    :param paddding:
    :param weight: weights of R, G, B channels. each element ranges from 0 to 1
    :return:

    '''

    # (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    template_dim = len(T.shape)

    # 2 dimensional template
    if template_dim == 2:
        h, w = T.shape
        T = normalize(T)  # normalize the template

        heatmap = np.zeros(
            ((I.shape[0] - h + paddding[0] + 1) // stride[0], (I.shape[1] - w + +paddding[1] + 1) // stride[0]))
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):

                Ir = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 0]
                Ig = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 1]
                Ib = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 2]

                # normalize the image part
                Ir = normalize(Ir)
                Ig = normalize(Ig)
                Ib = normalize(Ib)

                r = (Ir * T).sum()
                g = (Ig * T).sum()
                b = (Ib * T).sum()
                heatmap[i, j] = max(r, g, b)  # maximum value in three channels
                heatmap[i, j] = np.dot(weight, np.array([r, g, b]))  # weighted value over three channels

    else:  # template_dim == 3
        h, w, _ = T.shape

        # normalize the template
        Tr = normalize(T[:, :, 0])
        Tg = normalize(T[:, :, 1])
        Tb = normalize(T[:, :, 2])

        heatmap = np.zeros(
            ((I.shape[0] - h + paddding[0] + 1) // stride[0], (I.shape[1] - w + +paddding[1] + 1) // stride[0]))

        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                # normalize the image part
                Ir = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 0]
                Ig = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 1]
                Ib = I[i * stride[0]: i * stride[0] + h, j * stride[1]: j * stride[1] + w, 2]

                Ir = normalize(Ir)
                Ig = normalize(Ig)
                Ib = normalize(Ib)

                r = (Ir * Tr).sum()
                g = (Ig * Tg).sum()
                b = (Ib * Tb).sum()
                heatmap[i, j] = max(r, g, b)  # maximum value in three channels
                heatmap[i, j] = np.dot(weight, np.array([r, g, b]))  # weighted value over three channels

    # heatmap = np.random.random((n_rows, n_cols))

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, T, threshold=0.95, stride=(1, 1)):
    '''
    This function takes a numpy array <heatmap> and returns a list <bounding_boxes> and associated
    confidence scores.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    h, w = T.shape[0], T.shape[1]

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            # to avoid overlapped boxes
            if heatmap[i, j] > threshold and (
                    not output or (i * stride[0] > tl_row + 5 and j * stride[1] > tl_col + 5)):
                tl_row = i * stride[0]
                tl_col = j * stride[1]
                br_row = tl_row + h
                br_col = tl_col + w

                map_h = int(h / stride[0])
                map_w = int(w / stride[1])
                score = heatmap[i:i + map_h, j:j + map_w].sum() / (map_h * map_w)

                output.append([tl_row, tl_col, br_row, br_col, score])

    # '''
    # As an example, here's code that generates between 1 and 5 random boxes
    # of fixed size and returns the results in the proper format.
    # '''
    #
    # box_height = 8
    # box_width = 6
    #
    # num_boxes = np.random.randint(1, 5)
    #
    # for i in range(num_boxes):
    #     (n_rows, n_cols, n_channels) = np.shape(I)
    #
    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width
    #
    #     score = np.random.random()
    #
    #     output.append([tl_row, tl_col, br_row, br_col, score])
    #
    # '''
    # END YOUR CODE
    # '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    output = []

    # choose the template
    data_path = '../data/template_light'



    # get sorted list of files:
    file_names = sorted(os.listdir(data_path))

    # remove any non-JPEG files:
    file_names = [f for f in file_names if '.jpg' in f and 'template' in f]

    ############# enhanced version:
    # 1. using multiple template stages and combine the results
    # 2. averages heatmaps from three different filters
    for file_name in file_names:
        T = Image.open(os.path.join(data_path, file_name))

        T = np.asarray(T)
        heatmap = compute_convolution(I, T)

        # weakened version: using just one red channel
        T = np.asarray(T[:,:,0])
        heatmap = compute_convolution(I, T, weight=[1,0,0])

        output = output + predict_boxes(heatmap,T)

    # ############# weakened version:
    # # 1. using one random template stages
    # # 2. compute heatmaps from just red channel
    # T = Image.open(os.path.join(data_path, random.choice(file_names)))
    # T = np.asarray(T)
    # heatmap = compute_convolution(I, T[:,:,0], weight=[1,0,0])

    output = predict_boxes(heatmap,T)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    print(i)
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
    json.dump(preds_train, f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test.json'), 'w') as f:
        json.dump(preds_test, f)
