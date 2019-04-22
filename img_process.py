from PIL import Image
import cv2
import numpy as np
import os
import sys
import shutil
import math

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

def blur(img_file):
    img = cv2.imread(img_file)
    blur_img = cv2.GaussianBlur(img,(21,21),0)
    out_path = './output/blur.png'
    cv2.imwrite( out_path, blur_img )
    return out_path

def threshold(img_file):
    img = cv2.imread(img_file)
    threshold_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    out_path = './output/threshold.png'
    cv2.imwrite( out_path, threshold_img )
    return out_path

def image_mean(dir):
    img_paths = [];
    for image in os.listdir(dir):
        if (image == '.DS_Store'): continue
        img_paths.append(os.path.join(dir,image))

    avg_img = cv2.imread(img_paths[0]) * (1/len(img_paths))
    for image in img_paths[1:]:
        avg_img = cv2.add(avg_img, cv2.imread(image) * (1/len(img_paths)))
    return avg_img

def pre_process(dir):
    img_paths = [];
    for image in os.listdir(dir):
        if (image == '.DS_Store'): continue
        img_paths.append(os.path.join(dir,image))

    # CREATE PARENT DIRECTORY FOR PRE-PROCESSED IMAGES
    out_path = './pre_processed_images/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # SORT
    img_paths.sort()

    # CREATE DIRECTORIES FOR BINS
    total_num_imgs = len(img_paths)
    num_imgs_per_bin = 100
    num_bins = math.ceil(float(total_num_imgs) / float(num_imgs_per_bin))
    if num_bins == 0 and total_num_imgs > 0:
        num_bins = 1
    curr_image_index = 0

    dir_tokens = dir.split("/")
    dir_name = dir_tokens[-2]
    for i in range(num_bins):
        bin_path = out_path + dir_name + "_" + "bin_" + str(i+1) + "/"
        os.mkdir(bin_path)
        curr_num_imgs = 0
        while (curr_num_imgs < num_imgs_per_bin or i == num_bins-1):
            if (curr_image_index >= len(img_paths)):
                break
            img = cv2.imread(img_paths[curr_image_index])
            # HIST EQ
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            hist_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            # GREYSCALE
            greyscale_output = cv2.cvtColor(hist_output, cv2.COLOR_BGR2GRAY)
            # GAUSISAN BLUR
            blur_output = cv2.GaussianBlur(greyscale_output,(21,21),0)
            # ADAPTIVE THRESHOLD
            adaptive_threshold = cv2.adaptiveThreshold(blur_output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 105, 10)
            # SAVE
            cv2.imwrite( bin_path + img_paths[curr_image_index].split("/")[-1], adaptive_threshold )

            curr_num_imgs+=1
            curr_image_index+=1

        os.rename(bin_path, out_path + dir_name + "_" + "bin" + str(curr_image_index - curr_num_imgs) + "to" + str(curr_image_index-1))
    return out_path

def process(dir):
    out_path = './processed_output/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    pre_process_paths = []
    process_paths = []
    for subdir in os.listdir(dir):
        if (subdir == '.DS_Store'): continue
        path = os.path.join(dir,subdir)
        pre_process_paths.append(path + "/")
        process_path = out_path + path.split("/")[-1] + "/"
        process_paths.append(process_path)
        os.mkdir(process_path)

    for i in range(len(pre_process_paths)):
        dir = pre_process_paths[i]
        avg_img = image_mean(dir)
        threshold_img = cv2.threshold(avg_img, 127, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite( process_paths[i] + process_paths[i].split("/")[-2] + '_finalMask.png', threshold_img )

def blur_detection(img_file):
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 50;

    # # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite( "./output/keypoints.jpg", im_with_keypoints );

# histogram_equalization('./test/393408722.jpg')
# greyscale('./test/393408722.jpg')
# blur('./test/393408722.jpg')
# mean_img = image_mean('./test/')
# cv2.imwrite( "./output/avg_img.png", mean_img );

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please supply a directory of images. Usage: python3 img_process.py [img_dir_path]")
        exit(0)

    dir = sys.argv[1]
    if dir[-1] != "/":
        dir += "/"
    pre_processed_images_path = pre_process(dir)
    process = process(pre_processed_images_path)
