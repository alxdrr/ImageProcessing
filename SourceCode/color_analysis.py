from colorama import Fore, Style
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis, entropy
import cv2

def img_to_pxl(img):
    # create pixel matrix
    pixel_matrix = np.array(img)

    # set output to print only a few matrix data
    np.set_printoptions(edgeitems=5)

    # print pixel matrix
    print(Fore.GREEN,"Pixel Matrix \t: \n", Style.RESET_ALL, pixel_matrix, "\n================================================", sep="")

def histogram(img):
    print(Fore.GREEN,"Histogram \t:", Style.RESET_ALL)
    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create histogram
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # Plot histogram
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

def first_order_statistic(img):
    # Calculate mean
    mean = np.mean(img)

    # Calculate skewness
    skewness = skew(img.ravel())

    # Calculate variance
    variance = np.var(img)

    # Calculate kurtosis
    kurt = kurtosis(img.ravel())

    # Calculate entropy
    hist, _ = np.histogram(img, bins=256)
    hist = hist.astype(float) / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + (hist == 0)))

    # Print results
    print(Fore.GREEN,"First Order Statistic \t:", Style.RESET_ALL)
    print("Mean:", mean)
    print("Skewness:", skewness)
    print("Variance:", variance)
    print("Kurtosis:", kurt)
    print("Entropy:", entropy)