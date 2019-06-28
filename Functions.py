import numpy as np
import copy
import math
import random
import threading
import time
import PIL
import scipy
import scipy.ndimage as ndi
from scipy import linalg
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from six.moves import range
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from six.moves import xrange

def pad_distort_im_fn(x,output_size=None):
    """ Zero pads an image to 40X40, and distort it
    
    Examples
    -----------
    x = pad_distort_im_fn(X_train[0])
    print(x, x.shape, x.max())
    tl.vis.save_image(x, '_xd.png')
    tl.vis.save_image(X_train[0], '_x.png')
    """
    
    assert len(output_size) == 3
    b = np.zeros(output_size)
    height = output_size[0]
    width = output_size[1]
    o = int((height-28)/2)
    w = int((width-28)/2)
    b[o:o+28, w:w+28] = x
    x = b
    x = rotation(x, rg=30, is_random=True, fill_mode='nearest')
    x = shear(x, 0.05, is_random=True, fill_mode='nearest')
    x = shift(x, wrg=0.25, hrg=0.25, is_random=True, fill_mode='nearest')
    x = zoom(x, zoom_range=(0.95, 1.05))
    return x

def apply_fn(x, fn,output_size):
    return np.array([fn(xi,output_size) for xi in x])

def pad_distort_ims_fn(X,output_size=None):
    """ 
    Zero pads images to 40x40, and distort them. 
    """
    X_40 = []
    for X_a, _ in tqdm(minibatches(X, X, 50, shuffle=False)):
        X_40.extend(apply_fn(X_a, fn=pad_distort_im_fn,output_size=output_size))
    X_40 = np.asarray(X_40)
    return X_40


def rotation(
        x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0., order=1
):
    """Rotate an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    rg : int or float
        Degree to rotate, usually 0 ~ 180.
    is_random : boolean
        If True, randomly rotate. Default is False
    row_index col_index and channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : str
        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__
    cval : float
        Value used for points outside the boundaries of the input if mode=`constant`. Default is 0.0
    order : int
        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__

    Returns
    -------
    numpy.array
        A processed image.

    Examples
    ---------
    >>> x --> [row, col, 1]
    >>> x = tl.prepro.rotation(x, rg=40, is_random=False)
    >>> tl.vis.save_image(x, 'im.png')

    """
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


def affine_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    """Return transformed images by given an affine matrix in Scipy format (x is height).

    Parameters
    ----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    transform_matrix : numpy.array
        Transform matrix (offset center), can be generated by ``transform_matrix_offset_center``
    channel_index : int
        Index of channel, default 2.
    fill_mode : str
        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__
    cval : float
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
    order : int
        The order of interpolation. The order has to be in the range 0-5:
            - 0 Nearest-neighbor
            - 1 Bi-linear (default)
            - 2 Bi-quadratic
            - 3 Bi-cubic
            - 4 Bi-quartic
            - 5 Bi-quintic
            - `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__

    Returns
    -------
    numpy.array
        A processed image.

    Examples
    --------
    >>> M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)
    >>> M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)
    >>> M_combined = M_shear.dot(M_zoom)
    >>> transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, h, w)
    >>> result = tl.prepro.affine_transform(image, transform_matrix)

    """
    # transform_matrix = transform_matrix_offset_center()
    # asdihasid
    # asd

    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [
        ndi.interpolation.affine_transform(
            x_channel, final_affine_matrix, final_offset, order=order, mode=fill_mode, cval=cval
        ) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def shear(
        x, intensity=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.,
        order=1
):
    """Shear an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    intensity : float
        Percentage of shear, usually -0.5 ~ 0.5 (is_random==True), 0 ~ 0.5 (is_random==False),
        you can have a quick try by shear(X, 1).
    is_random : boolean
        If True, randomly shear. Default is False.
    row_index col_index and channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : str
        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__
    cval : float
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.
    order : int
        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__

    Returns
    -------
    numpy.array
        A processed image.

    References
    -----------
    - `Affine transformation <https://uk.mathworks.com/discovery/affine-transformation.html>`__

    """
    if is_random:
        shear = np.random.uniform(-intensity, intensity)
    else:
        shear = intensity
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x

def transform_matrix_offset_center(matrix, y, x):
    """Convert the matrix from Cartesian coordinates (the origin in the middle of image) to Image coordinates (the origin on the top-left of image).

    Parameters
    ----------
    matrix : numpy.array
        Transform matrix.
    x and y : 2 int
        Size of image.

    Returns
    -------
    numpy.array
        The transform matrix.

    Examples
    --------
    - See ``tl.prepro.rotation``, ``tl.prepro.shear``, ``tl.prepro.zoom``.
    """
    o_x = (x - 1) / 2.0
    o_y = (y - 1) / 2.0
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def shift(
        x, wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.,
        order=1
):
    """Shift an image randomly or non-randomly.

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Percentage of shift in axis x, usually -0.25 ~ 0.25.
    hrg : float
        Percentage of shift in axis y, usually -0.25 ~ 0.25.
    is_random : boolean
        If True, randomly shift. Default is False.
    row_index col_index and channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    fill_mode : str
        Method to fill missing pixel, default `nearest`, more options `constant`, `reflect` or `wrap`, see `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__
    cval : float
        Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0.
    order : int
        The order of interpolation. The order has to be in the range 0-5. See ``tl.prepro.affine_transform`` and `scipy ndimage affine_transform <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.affine_transform.html>`__

    Returns
    -------
    numpy.array
        A processed image.

    """
    h, w = x.shape[row_index], x.shape[col_index]
    if is_random:
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
    else:
        tx, ty = hrg * h, wrg * w
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = affine_transform(x, transform_matrix, channel_index, fill_mode, cval, order)
    return x


# zoom
def zoom(x, zoom_range=(0.9, 1.1), flags=None, border_mode='constant'):
    """Zooming/Scaling a single image that height and width are changed together.

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    zoom_range : float or tuple of 2 floats
        The zooming/scaling ratio, greater than 1 means larger.
            - float, a fixed ratio.
            - tuple of 2 floats, randomly sample a value as the ratio between 2 values.
    border_mode : str
        - `constant`, pad the image with a constant value (i.e. black or 0)
        - `replicate`, the row or column at the very edge of the original is replicated to the extra border.

    Returns
    -------
    numpy.array
        A processed image.

    """
    zoom_matrix = affine_zoom_matrix(zoom_range=zoom_range)
    h, w = x.shape[0], x.shape[1]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = affine_transform_cv2(x, transform_matrix, flags=flags, border_mode=border_mode)
    return x


def affine_transform_cv2(x, transform_matrix, flags=None, border_mode='constant'):
    """Return transformed images by given an affine matrix in OpenCV format (x is width). (Powered by OpenCV2, faster than ``tl.prepro.affine_transform``)

    Parameters
    ----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    transform_matrix : numpy.array
        A transform matrix, OpenCV format.
    border_mode : str
        - `constant`, pad the image with a constant value (i.e. black or 0)
        - `replicate`, the row or column at the very edge of the original is replicated to the extra border.

    Examples
    --------
    >>> M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)
    >>> M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=0.8)
    >>> M_combined = M_shear.dot(M_zoom)
    >>> result = tl.prepro.affine_transform_cv2(image, M_combined)
    """
    rows, cols = x.shape[0], x.shape[1]
    if flags is None:
        flags = cv2.INTER_AREA
    if border_mode is 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode is 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    else:
        raise Exception("unsupport border_mode, check cv.BORDER_ for more details.")
    return cv2.warpAffine(x, transform_matrix[0:2,:], \
            (cols,rows), flags=flags, borderMode=border_mode)



def minibatches(inputs=None, targets=None, batch_size=None, allow_dynamic_batch_size=False, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    targets : numpy.array
        The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is not a multiple of batch_size, this may result in unexpected behaviour if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=2, shuffle=False):
    >>>     print(batch)
    (array([['a', 'a'], ['b', 'b']], dtype='<U1'), array([0, 1]))
    (array([['c', 'c'], ['d', 'd']], dtype='<U1'), array([2, 3]))
    (array([['e', 'e'], ['f', 'f']], dtype='<U1'), array([4, 5]))

    Notes
    -----
    If you have two inputs and one label and want to shuffle them together, e.g. X1 (1000, 100), X2 (1000, 80) and Y (1000, 1), you can stack them together (`np.hstack((X1, X2))`)
    into (1000, 180) and feed to ``inputs``. After getting a batch, you can split it back into X1 and X2.

    """
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        if (isinstance(inputs, list) or isinstance(targets, list)) and (shuffle ==True):
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt], [targets[i] for i in excerpt]
        else:
            yield inputs[excerpt], targets[excerpt]
            
def affine_zoom_matrix(zoom_range=(0.8, 1.1)):
    """Create an affine transform matrix for zooming/scaling an image's height and width.
    OpenCV format, x is width.

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    zoom_range : float or tuple of 2 floats
        The zooming/scaling ratio, greater than 1 means larger.
            - float, a fixed ratio.
            - tuple of 2 floats, randomly sample a value as the ratio between these 2 values.

    Returns
    -------
    numpy.array
        An affine transform matrix.

    """

    if isinstance(zoom_range, (float, int)):
        scale = zoom_range
    elif isinstance(zoom_range, tuple):
        scale = np.random.uniform(zoom_range[0], zoom_range[1])
    else:
        raise Exception("zoom_range: float or tuple of 2 floats")

    zoom_matrix = np.array([[scale, 0, 0], \
                            [0, scale, 0], \
                            [0, 0, 1]])
    return zoom_matrix

def spatial_transformer_network(input_fmap, theta,out_dims=None,**kargs):
    """
    The layer is composed of 3 elements:
   	- localization_net 
    	- affine_grid_generator
    	- bilinear_sampler
    Input
    -------
    - input_fmap:  output of the previous layer (B,H,W,C)
    - theta: affine transform tensor of shape (B,6)

    Return
    -------
    - out_fmap: transformed input feature map (B,H,W,C)
    """

    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]

    theta = tf.reshape(theta,[B,2,3])

    if out_dims:
        out_dims = out_dims[0]
        out_W = out_dims[1]
        out_H = out_dims[2]
        batch_grids = affine_grid_generator(out_H,out_W,theta)
    else:
        batch_grids = affine_grid_generator(H,W,theta)

    x_s = batch_grids[:,0,:,:]
    y_s = batch_grids[:,1,:,:]

    out_fmap = bilinear_sampler(input_fmap,x_s,y_s)

    return out_fmap

def affine_grid_generator(height,width,theta):
    """
    This function returns a sampling grid, which when used with the binlinear sampler
    on the input feature map.

    Input
    ------
    - height
    - width
    - theta  of shape (num_batch,2,3)

    Returns
    ------
    - normalized grid (-1,1) of shape (B,2,H,W)
    """
    num_batch = tf.shape(theta)[0]
    
    x = tf.linspace(-1.,1.,width)
    y = tf.linspace(-1.,1.,height)
    x_t,y_t = tf.meshgrid(x,y)

    #flatten
    x_t_flat = tf.reshape(x_t,[-1])
    y_t_flat = tf.reshape(y_t,[-1])

    #reshape to [x_t,y_t,1] 
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat,y_t_flat,ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid,axis=0)
    sampling_grid = tf.tile(sampling_grid,[num_batch,1,1])

    # cast to float32 
    theta = tf.cast(theta,'float32')
    sampling_grid = tf.cast(sampling_grid,'float32')

    # transform the sampling grid
    batch_grids = tf.matmul(theta,sampling_grid)

    # reshape to (num_batch,H,W,2)
    batch_grids = tf.reshape(batch_grids,[num_batch,2,height,width])

    return batch_grids


def bilinear_sampler(img,x,y):
    """
    Input
    ------
    - img: batch of images in (B,H,W,C) layout
    - grid: x,y which is the output of affine_grid_generator

    Returns
    ------
    - out: interpolated iamges according to grids. Same size as grid
    
    """

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H-1,'int32')
    max_x = tf.cast(W-1,'int32')
    zero = tf.zeros([],dtype='int32')

    # Rescale x and y to [0,W-1/H-1]
    x = tf.cast(x,'float32')
    y = tf.cast(y,'float32')
    x = 0.5 * ((x+1.0)*tf.cast(max_x-1,'float32'))
    y = 0.5 * ((y+1.0)*tf.cast(max_y-1,'float32'))

    # grab 4 nearest corner points for each (x_i,y_i)
    x0 = tf.cast(tf.floor(x),'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y),'int32')
    y1 = y0 + 1

    # clip to range [0.H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0,zero,max_x)
    x1 = tf.clip_by_value(x1,zero,max_x)
    y0 = tf.clip_by_value(y0,zero,max_y)
    y1 = tf.clip_by_value(y1,zero,max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img,x0,y0)
    Ib = get_pixel_value(img,x0,y1)
    Ic = get_pixel_value(img,x1,y0)
    Id = get_pixel_value(img,x1,y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0,'float32')
    x1 = tf.cast(x1,'float32')
    y0 = tf.cast(y0,'float32')
    y1 = tf.cast(y1,'float32')

    # calculate deltas
    wa = (x1-x)*(y1-y)
    wb = (x1-x)*(y-y0)
    wc = (x-x0)*(y1-y)
    wd = (x-x0)*(y-y0)

    # add dimension for addition because of the channel dimension
    wa = tf.expand_dims(wa,axis=3)
    wb = tf.expand_dims(wb,axis=3)
    wc = tf.expand_dims(wc,axis=3)
    wd = tf.expand_dims(wd,axis=3)

    out = tf.add_n([wa*Ia,wb*Ib,wc*Ic,wd*Id])

    return out

def get_pixel_value(img,x,y):
    '''
    Input
    ------
    - img: tensor of shape (B,H,W,C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    '''
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0,batch_size)
    batch_idx = tf.reshape(batch_idx,(batch_size,1,1))
    b = tf.tile(batch_idx,(1,height,width))
    #channel_id = tf.zeros_like(b)  
    ## added one index for channel, if I don't specify
    ## the channel_id, then the pixel value with channel will be returned
    ## An extra dimension will be returned.

    indices = tf.stack([b,y,x],3)

    return tf.gather_nd(img,indices)

class keras_tranformer_layer(tf.keras.layers.Layer):
    def __init__(self,output_dims=None,**kwargs):
        self.output_dims = output_dims
        super(keras_tranformer_layer,self).__init__(**kwargs)

    def build(self,input_shape):
        super(keras_tranformer_layer,self).build(input_shape)

    def call(self,inputs):
        input_fmap, theta = inputs

        # grab input dimensions
        B = tf.shape(input_fmap)[0]
        H = tf.shape(input_fmap)[1]
        W = tf.shape(input_fmap)[2]

        theta = tf.reshape(theta,[B,2,3])

        '''
        The output dimension is determined from the output of affine_grid_generator.
        If out_dims is specified, then out_fmap's dimensions are the same as the
        out_dims, if it is not specified, then its dimensions are determined by the
        input feature map.
        '''

        if self.output_dims:
            out_W = self.output_dims[0]
            out_H = self.output_dims[1]
            batch_grids = affine_grid_generator(out_H,out_W,theta)
        else:
            batch_grids = affine_grid_generator(H,W,theta)

        x_s = batch_grids[:,0,:,:]
        y_s = batch_grids[:,1,:,:]

        out_fmap = bilinear_sampler(input_fmap,x_s,y_s)

        return out_fmap

    def compute_output_shape(self,input_shape):
        assert isinstance(input_shape,list)
        shape_fmap, shape_theta = input_shape
        if self.output_dims is not None:
            print(self.output_dims)
            return (shape_fmap[0],self.output_dims)
        else:
            print(shape_fmap)
            return shape_fmap

        
## ----------------- V2 Transformer
def transformer_V2(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


class keras_tranformer_layer_V2(tf.keras.layers.Layer):
    def __init__(self,output_dims=None,**kwargs):
        self.output_dims = output_dims
        super(keras_tranformer_layer_V2,self).__init__(**kwargs)

    def build(self,input_shape):
        super(keras_tranformer_layer_V2,self).build(input_shape)

    def call(self,inputs):
        input_fmap, theta = inputs

        # grab input dimensions
        B = tf.shape(input_fmap)[0]
        H = tf.shape(input_fmap)[1]
        W = tf.shape(input_fmap)[2]
        channel = tf.shape(input_fmap)[3]

        theta = tf.reshape(theta,[B,2,3])

        '''
        The output dimension is determined from the output of affine_grid_generator.
        If out_dims is specified, then out_fmap's dimensions are the same as the
        out_dims, if it is not specified, then its dimensions are determined by the
        input feature map.
        '''

        if self.output_dims:
            out_W = self.output_dims[0]
            out_H = self.output_dims[1]
            out_channel = self.output_dims[2]
            #batch_grids = affine_grid_generator(out_H,out_W,theta)
            return transformer_V2(input_fmap,theta,[out_H,out_W,out_channel])
        else:
            #batch_grids = affine_grid_generator(H,W,theta)
            return transformer_V2(input_fmap,theta,[H,W,channel])

    def compute_output_shape(self,input_shape):
        assert isinstance(input_shape,list)
        shape_fmap, shape_theta = input_shape
        if self.output_dims is not None:
            print(self.output_dims)
            return (shape_fmap[0],self.output_dims)
        else:
            print(shape_fmap)
            return shape_fmap



## ----------------------------------------------------------
## Code part for Thin Plate Spline interpolation
## --------------------------------------------------------
# Refer to https://github.com/WarBean/tps_stn_pytorch
# The original code is in Pytorch

def U_matrix(input_points,control_points):
    '''
    Function kernel_matrix will calculate the Euclidean distance between points,
    arrange them in the distance matrix.

    Input
    ---------------
        input_points:  -- The points tensor whose distance from control points will be
        calculated. The shape should be (N,2) where 2 is corresponding to x and y.
    
    Output
    ----------------
        output_points:  -- Control points tensor, it can be any points other than only
        control points. The shape should be (M,2) shere 2 is corresponding to x and y.

    Return 
    -----------------
        kernel_matrx  -- r^2log||r|| items of each distance,||r|| is the Euclidean
        distance between input point and control point.
    '''

    N = tf.shape(input_points)[0]
    M = tf.shape(control_points)[0]
    
    input_points_tensor = tf.expand_dims(input_points,axis=1)
    input_points_tensor = tf.tile(input_points_tensor,[1,M,1])
    control_points_tensor = tf.expand_dims(control_points_tensor,axis=0)
    control_points_tensor = tf.tile(control_points_tensor,[N,1,1])


    pairwise_diff = input_points_tensor - control_points_tensor
    pairwise_diff_square = tf.square(pairwise_diff)
    pairwise_dist = tf.reduce_sum(pairwise_diff_square,axis=-1)
    U_matrix = 0.5*pairwise_dist*tf.log(pairwise_dist)
    U_matrix = tf.where(tf.is_nan(U_matrix,tf.zeros_like(U_matrix), \
                U_matrix)

    return U_matrix

def TPSGridGen(target_height,target_width,target_control_points,source_control_points):
    '''
    Thin Plate Spline source grid generation based on the target grid and the source
    control point parameters that are passed to current layer.

    Inputs
    ----------------------
        target_height   -- Moving grid height
        target_width    -- Moving grid width
        target_control_points  -- Temp moving grid control points


    Outputs
    ----------------------
        source_coordinate    -- The source_coordinates of the grid points in the
        moving/target image

    '''

    assert tf.rank(target_control_points) == 2
    assert tf.shape(target_control_points)[1] == 2
    
    N = tf.shape(target_control_points)[0]
    target_control_points = tf.cast(target_control_points,dtype=tf.float32)
    U_target_control = U_matrix(target_control_points,target_control_points)
    




























