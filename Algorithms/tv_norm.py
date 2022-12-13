import numpy as np

"""Taken from https://github.com/cryoem/eman2/blob/master/examples/e2tvrecon2d.py"""


def tv_norm(image, norm='l1'):
    """Compute the mean (isotropic) TV norm of an image"""
    image = image.astype('float32')
    image /= np.max(image)

    grad_x1 = np.diff(image, axis=0)
    grad_x2 = np.diff(image, axis=1)

    performance = np.sqrt(grad_x1[:, :-1] ** 2 + grad_x2[:-1, :] ** 2).sum()

    if norm == 'l0':
        performance /= np.linalg.norm(np.reshape(image, -1), ord=0)
    elif norm == 'l1':
        performance /= np.linalg.norm(np.reshape(image, -1), ord=1)
    elif norm == 'l2':
        performance /= np.linalg.norm(np.reshape(image, -1), ord=2)
    elif norm == 'l21':
        performance /= np.linalg.norm(image, ord=2, axis=1).sum()

    return performance


def tv_l0_norm(img):
    """Compute the (isotropic) TV norm of a 2D image"""
    grad_x1 = np.diff(img, axis=0)
    grad_x2 = np.diff(img, axis=1)
    return (grad_x1[:, :-1] ** 2 + grad_x2[:-1, :] ** 2 > 0).mean()


def tv_norm_anisotropic(img):
    """Compute the anisotropic TV norm of an image"""
    grad_x1 = np.diff(img, axis=0)
    grad_x2 = np.diff(img, axis=1)
    return np.abs(grad_x1[:, :-1]).sum() + np.abs(grad_x2[:-1, :]).sum()
