import Image
from StringIO import StringIO
from tensorflow import Summary
from numpy import uint8

def log_scalar(tag, value):
    """
    log_scalar

    Logs a scalar.
    Gist code adapted from gyglim/tensorboard_logging.py
    """
    return Summary(value=[Summary.Value(tag=tag, simple_value=value)])

def log_images(tag, images, tagsuffix=''):
    """
    log_images

    Logs a list of images.
    Gist code adapted from gyglim/tensorboard_logging.py
    """
    def convert_to_uint8(img):
        return uint8(img * 255)

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
