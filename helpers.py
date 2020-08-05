import os
import glob
import matplotlib.image as mpimg


def load_dataset(image_dir):
    im_list = []
    image_types = ["red", "yellow", "green"]
    for im_type in image_types:
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            im = mpimg.imread(file)
            if im is not None:
                im_list.append((im, im_type, file))
    return im_list


