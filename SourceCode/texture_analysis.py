from skimage.feature import graycomatrix, graycoprops
from colorama import Fore, Style
import numpy as np
from matplotlib import pyplot as plt


def GLCM(img):
    # calculate GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # calculate contrast, correlation, energy, and homogeneity
    contrast = graycoprops(glcm, 'contrast')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0][0]
    asm = graycoprops(glcm, 'ASM')[0][0]

    # print results
    print(Fore.GREEN, "GLCM \t:", Style.RESET_ALL)
    print("Contrast:", contrast)
    print("Correlation:", correlation)
    print("Energy:", energy)
    print("Homogeneity:", homogeneity)
    print("Dissimilarity:", dissimilarity)
    print("ASM:", asm)


def txt_histogram(img):
    # calculate GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    print(Fore.GREEN,"Texture Histogram \t:", Style.RESET_ALL)

    # calculate Haralick Texture features
    features = ['contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM']
    ht = np.hstack([graycoprops(glcm, prop).ravel() for prop in features])

    # generate histogram of Haralick Texture features
    hist, _ = np.histogram(ht, bins=256)

    # plot histogram
    plt.bar(range(len(hist)), hist)
    plt.title('Haralick Texture Histogram')
    plt.xlabel('Texture Feature Values')
    plt.ylabel('Frequency')
    plt.show()

def second_order_statistic(img):
    # calculate co-occurrence matrix with distance=1 and angle=0 degrees
    glcm = graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)
    print(Fore.GREEN,"Second Order Statistic \t:", Style.RESET_ALL)

    print(Fore.BLUE, 'Co-occurrence matrix: \n',Style.RESET_ALL, glcm[:, :, 0, 0])

    # calculate texture features
    asm = graycoprops(glcm, 'ASM')[0, 0]
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    variance = graycoprops(glcm, 'energy')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    idm = 1 / (1 + dissimilarity)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    # print texture features
    print('Angular Second Moment (ASM):', asm)
    print('Contrast:', contrast)
    print('Correlation:', correlation)
    print('Variance:', variance)
    print('Inverse Difference Moment (IDM):', idm)
    print('Entropy:', entropy)