from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def transform(image):
    return image / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, image_path):
    images = inverse_transform(images) * 255
    images = np.concatenate(np.squeeze(images))

    return plt.imsave(image_path, images)


def gpu_split(ims, n_gpu, batch_size):
    split_dims = [0] * n_gpu
    drawn_batch_size = tf.shape(ims)[0]
    drawn_batch_size -= - n_gpu
    for i in range(n_gpu):
        split_dims[i] = tf.maximum(0, tf.minimum(batch_size - 1, drawn_batch_size)) + 1
        drawn_batch_size -= batch_size

    return tf.split(ims, split_dims)


def merge(images, size):
    """
    merge image sequence (backward + forward)
    Args:
        images (numpy.ndarray): the array of images (backward + forward sequences),  shape (2*T, 80, 80, 3)
        size (list): (2,1), example value: [2,T]

    Returns:

    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        print(idx, np.shape(image))
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def reshape_patch(img_tensor, patch_size):
    """Reshape a 5D image tensor to a 5D patch tensor."""
    # print(f"adasd {np.shape(img_tensor)}")
    # assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [
        batch_size, seq_length, img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size, num_channels
    ])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    patch_tensor = np.reshape(b, [
        batch_size, seq_length, img_height // patch_size, img_width // patch_size,
                                patch_size * patch_size * num_channels
    ])
    return patch_tensor










