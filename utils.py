import numpy as np

def segmentation(img, n):
    '''
    This is a preprocessing function that will segment the images into segments with dimensions n x n.
    Parameters
    ----------
    img : numpy.ndarray
        The image to be segmented.
    n : int
        The dimension of the segments.
    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    '''

    N = img.shape[0] // n #the number of whole segments in the y-axis
    M = img.shape[1] // n #the number of whole segments in the x-axis

    ####
    # there are 4 cases
    #+------------+------------+------------+
    #| *n         | y segments | x segments |
    #+------------+------------+------------+
    #| N !=, M != | N+1        | M+1        |
    #+------------+------------+------------+
    #| N !=, M =  | N+1        | M          |
    #+------------+------------+------------+
    #| N =, M !=  | N          | M+1        |
    #+------------+------------+------------+
    #| N =, M =   | N          | M          |
    #+------------+------------+------------+
    ####
    if N*n != img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N+1, M+1, n, n), dtype=np.float32)
    elif N*n != img.shape[0] and M*n == img.shape[1]:
        segments = np.zeros((N+1, M, n, n), dtype=np.float32)
    elif N*n == img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N, M+1, n, n), dtype=np.float32)
    else:
        segments = np.zeros((N, M, n, n), dtype=np.float32)

    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])

    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,i*n:(i+1)*n]
            elif i == x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,-n:]
            elif i != x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,i*n:(i+1)*n]
            elif i == x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,-n:]

    segments = np.reshape(segments, newshape=((segments.shape[0]*segments.shape[1]), n, n))

    return segments