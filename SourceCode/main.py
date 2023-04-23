from color_analysis import *
from texture_analysis import *
from colorama import Fore, Style
from skimage import io
import glob
import os

# path to the folder containing the images
path = "AssetImage/*.jpg"

# looping for processing all images
for img_path in glob.glob(path):
    # get the file name
    file_name = os.path.basename(img_path)

    # read image
    img = io.imread(img_path)

    # print file name
    print(Fore.BLUE, "Image Name : ", file_name, Style.RESET_ALL)

    # color analysis process
    # convert image to pixel matrix
    img_to_pxl(img)

    # image histogram
    histogram(img)

    # first order statistic
    first_order_statistic(img)

    # texture analysis process
    # GLCM
    GLCM(img)

    # texture histogram
    txt_histogram(img)

    # first order statistic
    second_order_statistic(img)
    input("Press Enter to continue...")
