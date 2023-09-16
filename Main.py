# imports
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# cap = cv2.VideoCapture(1)
# print("asdsad")
# ret, frame = cap.read()
# cv2.imshow("frame", frame)
# cv2.imwrite("frame.png", frame)
# print("aadsadsad")
# cv2.waitKey(1)
# cap.release()

def encodeHex(color):
    b = color[0]
    g = color[1]
    r = color[2]
    hex = '#' + str(bytearray([r, g, b]).hex())
    print(hex)
    return hex


# using numpy's min max scaler to transform the data before finding knee
def simplifyData(img):
    mms = MinMaxScaler()
    mms.fit(img)
    return mms.transform(img)  # data transformed


# finding the knee/elbow of the graph
def elbowGraph(data_transformed, rangeNum):
    SumOfSquaredDistances = []

    x = range(1, rangeNum)
    for i in x:
        print(i)
        km = KMeans(n_clusters=i)
        km.fit(data_transformed)
        SumOfSquaredDistances.append(km.inertia_)
    kn = KneeLocator(x, SumOfSquaredDistances, curve='convex', direction='decreasing')

    print("Optimal Number of Clusters is:")
    print(kn.knee)
    plt.plot(x, SumOfSquaredDistances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig('ElbowGraph.png')
    plt.show()
    return kn.knee


def hexToRgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgbToHsv(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v

def findPos(contours, contourIndex):
    xPosList = []
    for contour in contours:
        xPosList.append(contour.x)

    propXCoord = contours[contourIndex].x
    xPosList.sort()

    return xPosList.index(propXCoord) + 1

### WIP CODE ###
# def closest(list, K):
#     return list[min(range(len(list)), key=lambda i: abs(list[i] - K))]

def closest(colors, color):
    colors = np.array(colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return smallest_distance


# # Define a function to calculate the Euclidean distance between two HSV colors
# def hsvDistance(color1, color2):
#
#     # Calculate Euclidean distance
#     return np.linalg.norm(color1 - color2)

# read image into range 0 to 1

def findClosestColor(rgbList, realcolor):
    disValList = []
    for color2 in rgbList:
        r = float(color2[0] - realcolor[0])
        g = float(color2[1] - realcolor[1])
        b = float(color2[2] - realcolor[2])
        disVal = math.sqrt(((abs(r)) ** 2) + ((abs(g)) ** 2) + ((abs(b)) ** 2))
        disValList.append(disVal)
    return disValList


# TO READ PRESAVED IMAGE
img = cv2.imread('pic5.png') / 255

# READING CAMERA INIT FRAME
# img = cv2.imread('FRAME .jpg') / 255
img = cv2.resize(img, (640, 480))

blurImg = cv2.GaussianBlur(img, (19, 19), 0)

# reshape images
h, w, c = blurImg.shape
img2 = blurImg.reshape(h * w, c)
# Resize the image

# max k value to check for
rangeNum = 21

# finding the k value
optimalNum = elbowGraph(simplifyData(img2), rangeNum)

# set number of colors
number = optimalNum

# clustering into KMeans
kmeans_cluster = cluster.KMeans(n_clusters=number)
kmeans_cluster.fit(img2)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

# need to scale back to range 0-255 and reshape
img3 = cluster_centers[cluster_labels].reshape(h, w, c) * 255.0
img3 = img3.astype('uint8')

cv2.imshow('reduced colors', img3)

# reshape img to 1 column of 3 colors
# -1 means figure out how big it needs to be for that dimension
img4 = img3.reshape(-1, 3)

# get the unique colors
colors, counts = np.unique(img4, return_counts=True, axis=0)
print(colors)
print("============")
print(counts)
unique = zip(colors, counts)

# creating color lists
hList = []
HSV = []
RGBList = []
HSVList = []
hexList = []

# converting color output and printing to hex and RGB
for i, uni in enumerate(unique):
    color = uni[0]
    count = uni[1]
    hexColor = encodeHex(color)
    hexList.append(hexColor)
    RGB = hexToRgb(hexColor)
    h, s, v = rgbToHsv(RGB)
    hList.append(h)
    HSVList.append([h, s, v])
    print("RGB:")
    print(RGB)
    r = RGB[0]
    g = RGB[1]
    b = RGB[2]
    RGBList.append([r, g, b])
    print("HSV:")
    print([h, s, v])
    print("===")

# set to false if teamColor is blue
isRed = True

if isRed:
    color = [255, 0, 0]
else:
    color = [0, 255, 0]

distanceList = findClosestColor(RGBList, color)

indexOfColor = (distanceList.index(min(distanceList)))

# finding bgr color
BGR = [RGBList[indexOfColor][2], RGBList[indexOfColor][1], RGBList[indexOfColor][0]]

print(RGBList[indexOfColor])
print(BGR)

# masking out the red / blue color
lower = np.array(BGR)
upper = np.array(BGR)
imgArr = np.array(img3)

mask = cv2.inRange(imgArr, lower, upper)
masked = cv2.bitwise_and(imgArr, imgArr, mask=mask)
result = imgArr - masked

# saving the masked images
cv2.imwrite('masked.png', result)
cv2.imwrite('masked2.png', masked)
cv2.imwrite('masked3.png', mask)

# blurring
mask = cv2.GaussianBlur(mask, (13, 13), 0)

# dilation
kernel = np.ones((7, 7), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=4)

# showing the mas
cv2.imshow("Mask", mask)

# finding the contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', img)

# making list of contour areas
areaList = []
for contour in contours:
    areaList.append(cv2.contourArea(contour))

cv2.drawContours(img, contours[areaList.index(max(areaList))], -1, (255, 0, 255), 12)

# highlight the prop
cv2.imshow("POSITION", img)

# wait for image to close
cv2.waitKey(0)
cv2.destroyAllWindows()
