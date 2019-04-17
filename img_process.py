from PIL import Image
import cv2
import numpy as np
import os

def greyscale(img_file):
    img = Image.open(img_file).convert('LA')
    out_path = './output/greyscale.png'
    img.save(out_path)
    return out_path

def contrast(img_file):
    img = Image.open(img_file)
    img.load()
    level = 100
    factor = (259 * (level+255)) / (255 * (259-level))
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            color = img.getpixel((x, y))
            new_color = tuple(int(factor * (c-128) + 128) for c in color)
            img.putpixel((x, y), new_color)
    out_path = './output/contrast.png'
    img.save(out_path)
    return out_path

def histogram_equalization(img_file):
    img = cv2.imread(img_file)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    out_path = './output/hist_eq.png'
    cv2.imwrite( out_path, img_output );
    return out_path

def image_mean(dir):
    img_paths = [];
    for image in os.listdir(dir):
        img_paths.append(os.path.join(dir,image))

    avg_img = cv2.imread(img_paths[0]) * (1/len(img_paths))
    for image in img_paths[1:]:
        avg_img = cv2.add(avg_img, cv2.imread(image) * (1/len(img_paths)))
    out_path = './output/avg_img.png'
    cv2.imwrite( out_path, avg_img )
    return out_path

def blur_detection(img_file):
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 50;

    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    #
    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite( "./output/keypoints.jpg", im_with_keypoints );

# Pre-process: histeq, greyscale, increase contrast
# Process: Average Images for Noise Removal
# Process: Create Binary Mask (apply erosion and dilation to reduce further noise)
# Todo: Erosion, Dilation, and Create Binary Mask

histeq_img = histogram_equalization('./test/393408722.jpg')
greyscale_img = greyscale(histeq_img)
image_mean('./test/')
#contrast_img = contrast(greyscale_img)
#blur_detection(greyscale_img)
