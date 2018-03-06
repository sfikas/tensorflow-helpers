"""

    G.Sfikas Oct 2017
    Based on gist code from gyglim/tensorboard_logging.py

"""

import numpy as np
import Image
from StringIO import StringIO
from tensorflow import Summary, HistogramProto
import matplotlib.pyplot as plt

def log_scalar(tag, value):
    """
    log_scalar
    Logs a scalar.
    """
    return Summary(value=[Summary.Value(tag=tag, simple_value=value)])

def log_colorimages(tag, images, tagsuffix=''):
    img = images
    s = StringIO()
    plt.imsave(s, img, format='png')
    img_sum = Summary.Image(encoded_image_string=s.getvalue(),
                               height=img.shape[0],
                               width=img.shape[1])
    return Summary(value=[Summary.Value(tag='%s%s' % (tag, tagsuffix), image=img_sum)])

def log_images(tag, images, tagsuffix=''):
    """
    log_images
    Logs a list of images.
    """
    def convert_to_uint8(img):
        return np.uint8(img * 255)

    if not type(images) == list:
        img = images
        s = StringIO()
        Image.fromarray(convert_to_uint8(img), mode='L').save(s, 'png')
        # Create an Image object
        img_res = Summary.Image(encoded_image_string=s.getvalue(),
                                height=img.shape[0],
                                width=img.shape[1],
                                   colorspace=1)
        return Summary(value=[Summary.Value(tag='%s%s' % (tag, tagsuffix), image=img_res)])
    else:
        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            Image.fromarray(convert_to_uint8(img), mode='L').save(s, 'png')
            img_sum = Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0], width=img.shape[1],
                                       colorspace=1) #https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/framework/summary.proto
            # Create a Summary value
            im_summaries.append(Summary.Value(tag='%s/%d%s' % (tag, nr, tagsuffix), image=img_sum))
        return Summary(value=im_summaries)

def log_histogram(tag, values, step, bins=1000):
    """
    log_histogram
    Logs the histogram of a list/vector of values.
    """
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return Summary(value=[Summary.Value(tag=tag, histo=hist)])

def log_vector(tag, values):
    """
    log_histogram
    Logs a vector of values.
    """
    values = np.array(values).flatten()

    # Fill fields of histogram proto
    hist = HistogramProto()
    hist.min = 0
    hist.max = len(values) - 1
    hist.num = len(values)
    hist.sum = float(np.sum(np.arange(hist.num)))
    hist.sum_squares = float(np.sum(np.arange(hist.num) ** 2))

    for idx, c in enumerate(values):
        hist.bucket_limit.append(idx)
        hist.bucket.append(c)

    return Summary(value=[Summary.Value(tag=tag, histo=hist)])
