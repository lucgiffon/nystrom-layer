"""
Module containing popular kernel functions using the keras API.
"""

from keras import backend as K
from keras.activations import tanh
import tensorflow as tf

def replace_nan(tensor):
    return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)


def keras_linear_kernel(args, normalize=True, tanh_activation=False):
    """
    Linear kernel:

    $k(x, y) = x^Ty$

    :param args: list of size 2 containing x and y
    :param normalize: if True, normalize the input with l2 before computing the kernel function
    :param tanh_activation: if True apply tanh activation to the output
    :return: The linear kernel between args[0] and args[1]
    """
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    result = K.dot(X, K.transpose(Y))
    if tanh_activation:
        return tanh(result)
    else:
        return result



def keras_chi_square_CPD(args, epsilon=None, tanh_activation=True, normalize=False):
    """
    Chi square kernel (equivalent to `additive_chi2_kernel` in scikit-learn):

    $k(x, y) = -Sum [(x - y)^2 / (x + y)]$

    :param args: list of size 2 containing x and y
    :param epsilon: very small value to add to the denominator so that we do not have zeros here
    :param tanh_activation: if True apply tanh activation to the output
    :param normalize: if True, normalize the input with l2 before computing the kernel function
    :return: The chi square kernel between args[0] and args[1]
    """
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    # the drawing of the matrix X expanded looks like a wall
    wall = K.expand_dims(X, axis=1)
    # the drawing of the matrix Y expanded looks like a floor
    floor = K.expand_dims(Y, axis=0)
    numerator = K.square((wall - floor))
    denominator = wall + floor
    if epsilon is not None:
        quotient = numerator / (denominator + epsilon)
    else:
        quotient = numerator / denominator
    quotient_without_nan = replace_nan(quotient)
    result = - K.sum(quotient_without_nan, axis=2)
    if tanh_activation:
        return tanh(result)
    else:
        return result


def keras_chi_square_CPD_exp(args, gamma, epsilon=None, tanh_activation=False, normalize=True):
    """
    Exponential chi square kernel (equivalent to `chi2_kernel` in scikit-learn):

    $k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])$

    :param args: list of size 2 containing x and y
    :param epsilon: very small value to add to the denominator so that we do not have zeros here
    :param tanh_activation: if True apply tanh activation to the output
    :param normalize: if True, normalize the input with l2 before computing the kernel function
    :return: The exponential chi square kernel between args[0] and args[1]
    """
    result = keras_chi_square_CPD(args, epsilon, tanh_activation, normalize)
    result *= gamma
    return K.exp(result)


def keras_rbf_kernel(args, gamma, normalize=True, tanh_activation=False):
    """
    Compute the rbf kernel between each entry of X and each line of Y.

    $(x, y, gamma) = exp(- (||x - y||^2 * gamma))$

    :param X: A tensor of size n times d
    :param Y: A tensor of size m times d
    :param gamma: The bandwith of the kernel
    :return: The RBF kernel between args[0] and args[1]
    """
    X = args[0]
    Y = args[1]
    if normalize:
        X = K.l2_normalize(X, axis=-1)
        Y = K.l2_normalize(Y, axis=-1)
    r1 = K.sum(X * X, axis=1)
    r1 = K.reshape(r1, [-1, 1])
    r2 = K.sum(Y * Y, axis=1)
    r2 = K.reshape(r2, [1, -1])
    result = K.dot(X, K.transpose(Y))
    result = r1 - 2 * result + r2
    result *= -gamma
    result = K.exp(result)
    if tanh_activation:
        return tanh(result)
    else:
        return result