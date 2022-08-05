import numpy as np
import h5py


def load_dataset(train_set_location, test_set_location):
    """
    Load dataset into 4 variables

    Args:
    train_set_location -- location of train set
    test_set_location -- location of test set

    Returns:
    train_set_x_orig --
    train_set_y_orig --
    test_set_x_orig --
    test_set_y_orig --
    classes --
    """

    train_dataset = h5py.File(train_set_location, "r")
    # These data sets are to be pre-preprocessed. (Thus, appended "_orig")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(test_set_location, "r")
    # These data sets are to be pre-preprocessed. (Thus, appended "_orig")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m/2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m,D))  # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))
    
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- Size of the w vector we want. (or number of parameters in this case)
    
    Returns:
    w -- Initialized vector of shape. (dim, 1)
    b -- Initialized scalar. (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0
    
    # For image inputs, w will be of shape (num_px * num_px * 3, 1)
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding.
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid.
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)