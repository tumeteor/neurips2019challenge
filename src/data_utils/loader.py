import cv2
import numpy as np
import logging
import h5py
import os
import datetime
import re
from src.utils import transform, inverse_transform

# Data Loader Functions
ORIGINAL_SHAPE = (495, 436)
# TILE_SHAPE = (124, 109)
# TARGET_SHAPE = (124, 109)

TILE_SHAPE = (62, 73)
TARGET_SHAPE = (62, 73)


def _salt_and_pepper_noise(image: object, noise_typ: object = "gauss") -> object:
    """
    add Gaussian noise to distort the high-frequency features (zero pixels)
    Args:
        image (numpy.ndarray): the target image

    Returns:
        numpy.ndarray: the image with random noise
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape  # one channel
        mean = 0
        var = 0.05
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def _resize2_square_keeping_aspect_ration(image, target, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    c = None if len(image.shape) < 3 else image.shape[2]
    if h == w:
        return cv2.resize(image, target, interpolation)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=image.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = image[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=image.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = image[:h, :w, :]
    return cv2.resize(mask, target, interpolation)


def resize_image(image, target):
    return cv2.resize(src=image, dsize=(target[1], target[0]), interpolation=cv2.INTER_AREA)


def crop_image(image):
    # npad is a tuple of (n_before, n_after) for each dimension
    # return np.pad(image, ((0, 1), (0, 2), (0, 0)), 'constant')  # (495, 436) -> (496, 438)
    return np.pad(image, ((0, 1), (0, 0), (0, 0)), 'constant')  # (495, 436) -> (496. 436)
    # return image[0:492, 0:436, :]


def to_tiles(im, M, N):
    return np.array([im[x:x + M, y:y + N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)])


def reconstruct_from_tiles(tiles):
    # n_tiles, n_seq, tile_h, tile_w, n_channel
    # (5, 48, 6, 62, 73, 3)
    # tiles = np.transpose(tiles, (1, 0, 2, 3, 4))
    re_img = np.zeros((496, 438, 3))
    for n, patch in enumerate(tiles):
        # eight patches per row
        row, col = divmod(n, 8)
        row_offset, col_offset = row * 62, col * 73
        row_slice = slice(row_offset, 62 + row_offset)
        col_slice = slice(col_offset, 73 + col_offset)
        re_img[row_slice, col_slice] = patch
    re_img = re_img.astype(np.uint8)
    return re_img[0:ORIGINAL_SHAPE[0], 0:ORIGINAL_SHAPE[1], :]


def post_process_pred(data):
    data = [[reconstruct_from_tiles(data[i, j, :, :, :, :])
             for j in range(data.shape[1])] for i in range(data.shape[0])]
    return np.array(data)


def extract_date(filename):
    match = re.search(r'\d{4}\d{2}\d{2}', filename)
    return datetime.strptime(match.group(), '%Y%m%d').date()


def chunk_data(file_names, city):
    # sort by date
    file_names.sort(key=lambda date: extract_date(date))
    file_chunks = chunks(file_names, 12)
    data = []
    for files in file_chunks:
        for file in files:
            fr = h5py.File(file, 'r')
            a_group_key = list(fr.keys())[0]
            _data = [i[:, :, :] for i in fr[a_group_key]]
            data.append(_data)
        data = np.concatenate(data)
        with h5py.File(f"/data/{city}_chunk/{file}", 'w') as fr_grp:
            fr_grp.create_dataset("array", data=data)


def load_data(file_path, indices=None, K=10, T=10, training=True, batch_size=1):
    """Load data for one test day, return as numpy array with normalized samples of each
        6 time steps in random order.

        Args.:
            file_path (str): file path of h5 file for one day
            indices (list): list with prediction times (as list indices in the interval [0, 288])
            K (int): the number of images look back
            T (int): the number of images look ahead

        Returns: numpy array of shape (5, 6, 3, 495, 436)
    """
    # load h5 file
    logging.info("load data: {}".format(file_path))
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key]
    data = np.array(data)

    # data = [crop_image(image=i) for i in data]
    # resize image to target size
    # data = [resize_image(image=i, target=ORIGINAL_SHAPE) for i in data]

    data = [to_tiles(_salt_and_pepper_noise(crop_image(im)) / 255., TILE_SHAPE[0], TILE_SHAPE[1]) for im in data]

    # add gaussian noise
    if training:
        data = [_salt_and_pepper_noise(image=i) for i in data]
    # identify test cases and split in samples of each length 3 time bins
    data = [data[y - K: y + T] for y in indices] if indices else [data[int(y) - K: y + T] for y in
                                                                  range(K, len(data) - T - 1)]
    data = np.stack(data, axis=0)
    logging.debug("data shape: {}".format(np.shape(data)))
    data = np.transpose(data, (0, 2, 1, 3, 4, 5))  # remove channels

    print("data shape: {}".format(np.shape(data)))
    # rescale and return data
    data = data.astype(np.float32)
    # only shuffles the array along the first axis of a multi-dimensional array.
    if training:
        np.random.shuffle(data)

    minibatch = list(chunks(data, batch_size))
    print(f"length of minibatch: {len(minibatch)}")

    date = return_date(file_path)
    return minibatch, np.array([date.weekday()] * 12).reshape(1, -1)  # hard code implicit batch size


from itertools import islice


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) - len(l) % n, n):
        yield l[i:i + n]


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def list_filenames(directory, excluded_dates=None):
    """Auxilliary function which returns list of file names in directory in random order,
        filtered by excluded dates.

        Args.:
            directory (str): path to directory
            excluded_dates (list): list of dates which should not be included in result list,
                e.g., ['2018-01-01', '2018-12-31']

        Returns: list
    """
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)

    # check if in excluded dates
    _excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates] if excluded_dates \
        else None
    filenames = [x for x in filenames if return_date(x) not in _excluded_dates] if excluded_dates else filenames
    return filenames


def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.

        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'

        Returns: date string, e.g., '2018-05-16'
    """

    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date
