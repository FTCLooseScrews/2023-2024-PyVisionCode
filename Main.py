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


### WIP CODE ###
# def closest(list, K):
#     return list[min(range(len(list)), key=lambda i: abs(list[i] - K))]

#
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

def findClosestColor(hsvList, realColor):
    distanceList = []

    for color in hsvList:
        dh = min(abs(color[0] - realColor[0]), 360 - abs(color[0] - realColor[0])) / 180.0
        ds = abs(color[1] - realColor[1]) / 255.0
        dv = abs(color[2] - realColor[2]) / 255.0
        distance = math.sqrt(dh * dh + ds * ds + dv * dv)
        distanceList.append(distance)

    return distanceList

def findClosestColor2(rgbList, realcolor):
    disValList = []
    for color2 in rgbList:
        r = float(color2[0] - realcolor[0])
        g = float(color2[1] - realcolor[1])
        b = float(color2[2] - realcolor[2])
        disVal = math.sqrt( ((abs(r))**2) + ((abs(g))**2) + ((abs(b))**2) )
        disValList.append(disVal)
    return disValList

# TO READ PRESAVED IMAGE
img = cv2.imread('pic1.jpg') / 255

# READING CAMERA INIT FRAME
# img = cv2.imread('FRAME .jpg') / 255

# reshape images
h, w, c = img.shape
img2 = img.reshape(h * w, c)

# max k value to check for
rangeNum = 21

# finding the k value
optimalNum = elbowGraph(simplifyData(img2), rangeNum)

# set number of colors
number = optimalNum

# cluseting into KMeans
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

red = True
if red:
    color = [255, 0, 0]
else:
    color = [0, 255, 0]

distanceList = findClosestColor2(RGBList, color)

indexOfColor = (distanceList.index(min(distanceList)))

# finding bgr color
BGR = [RGBList[indexOfColor][2], RGBList[indexOfColor][1], RGBList[indexOfColor][0] ]

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

cv2.waitKey(0)
cv2.destroyAllWindows()
