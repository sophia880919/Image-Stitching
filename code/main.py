import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg
from PIL import Image, ImageFilter
from glob import glob as glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=int, default=5, help="sift threshold")
parser.add_argument("--cutoff", type=int, default=0.0003, help="0.5 BF , 0.0003 #flann")
parser.add_argument("--x_thres_partition", type=int, default=5, help="if == inf means no threshold")
parser.add_argument("--y_thres_partition", type=int, default=20, help="if == 1 means no threshold")
parser.add_argument("--K", type=int, default=1000, help="threshold loop times")
parser.add_argument("--threshold_distance", type=int, default=3)
parser.add_argument("--overlap_radius", type=int, default=25)
parser.add_argument("--crop_deno", type=int, default=50)
parser.add_argument("--ALIGN", default=True, action='store_true')
parser.add_argument("--CROP", default=True, action='store_true')
parser.add_argument("--path", type=str, default="../data")
args = parser.parse_args()

def load_images_and_focal_lengths(source_dir):
    img_filenames = sorted(glob(os.path.join(source_dir, '*.jpg')))
    images = [cv2.imread(img_filename, 1) for img_filename in img_filenames]

    info_filename = glob(os.path.join(source_dir, 'info.txt'))
    focal_lengths = []
    with open(info_filename[0]) as f:
        for line in f:
            (image_name, image_focal_length) = line.strip().split(" ")
            focal_lengths.append(float(image_focal_length))
    return np.array(images), np.array(focal_lengths)

def cylindrical_projection(imgs, focal_lengths):
    num, height, width, _ = imgs.shape
    cylinder_projs = []
    for i,img in enumerate(imgs):
        cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)
        focal_length = focal_lengths[i]
        for y in range(-int(height/2), int(height/2)):
            for x in range(-int(width/2), int(width/2)):
                cylinder_x = focal_length*math.atan(x/focal_length)
                cylinder_y = focal_length*y/math.sqrt(x**2+focal_length**2)
                cylinder_x = int(round(cylinder_x + width/2))
                cylinder_y = int(round(cylinder_y + height/2))
                if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                    cylinder_proj[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
        print ("project image {} to cylinderical coordinate...".format(i))
        cylinder_projs.append(cylinder_proj)
    return cylinder_projs

def alingment(img, shifts):
    height, width = img.shape[:2]
    total_shift_x, total_shift_y = np.sum(shifts, axis=0)
    scatter_shift = None

    if total_shift_y < 0:
        scatter_shift = np.linspace(0,-1*total_shift_y , num=width, dtype=np.uint8)
    elif total_shift_y > 0:
        scatter_shift = np.linspace(0,-1*total_shift_y , num=width, dtype=np.uint8)
    print(scatter_shift.shape, img.shape)

    img_aligned = img.copy()
    for x in range(width):
        img_aligned[:,x] = np.roll(img[:,x], scatter_shift[x], axis=0)

    return img_aligned

def RANSAC(matched_pairs):
    matched_pairs = np.asarray(matched_pairs)
    if len(matched_pairs) > args.K:
        use_random = True
        K = args.K
    else:
        use_random = False
        K = len(matched_pairs)

    best_shift = []
    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        idx = int(np.random.random_sample()*len(matched_pairs)) if use_random else k
        sample = matched_pairs[idx] # pick one pair of matching features

        # calculate shift
        shift = sample[1] - sample[0] # next one - last one

        # calculate inliner points
        predicted_pt = matched_pairs[:,1] - shift  # next one - shift = predicted last one
        differences = matched_pairs[:,0] - predicted_pt # last one - predicted last one
        inliner = 0
        for diff in differences:
            if np.sqrt(np.square(diff).sum()) < args.threshold_distance: # 2-norm distance
                inliner = inliner + 1
        if inliner > max_inliner:
            max_inliner,best_shift = inliner,shift

    return list(best_shift)

def rectangling(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    upper, lower = [0, img.shape[0]]
    max_black_pixel_num_thres = img.shape[1]//args.crop_deno
    for y in range(img_thresh.shape[0]):
        if len(np.where(img_thresh[y] == 0)[0]) < max_black_pixel_num_thres:
            upper = y
            break
    for y in range(img_thresh.shape[0]-1, 0, -1):
        if len(np.where(img_thresh[y] == 0)[0]) < max_black_pixel_num_thres:
            lower = y
            break
    return img[upper:lower, :]

def image_matching(shift_list, image_set_size, height, width):
    shift_set = np.array(shift_list)
    img_list = []
    for i in range(image_set_size):
        img = cv2.imread(str(i)+'.jpg') # size = 450x300
        img_list.append(img)
    shift_x = 0
    shift_y = 0
    shift_y_max = -1*float("inf")
    shift_y_min = float("inf")
    for shift in shift_list:
        shift_x += shift[0]
        shift_y += shift[1]
        if shift_y<shift_y_min: shift_y_min=shift_y
        if shift_y>shift_y_max: shift_y_max=shift_y
    shift_acc=[]
    shift_sum = np.array([0,0])
    for shift in shift_set:
        shift_sum+=shift
        temp = shift_sum.copy()
        temp[0] = -1*temp[0]
        shift_acc.append(temp)

    new_img = np.zeros( (height+abs(shift_y_min)+abs(shift_y_max), width+abs(shift_x),3),dtype=np.uint8)
    new_h, new_w  = new_img.shape[:2]
    left_br_x, right_br_x = 0, 0
    for img_num,img in enumerate(img_list):
        bl_r = args.overlap_radius
        for i,new_i in enumerate(range(shift_acc[img_num][1]+abs(shift_y_min), height+shift_acc[img_num][1]+abs(shift_y_min))):
            if img_num == 0:
                left_br_x = 0
                right_br_x = ((shift_acc[img_num][0]+width) + shift_acc[img_num+1][0])/2
                for j,new_j in enumerate(range(0, int(right_br_x-bl_r))): # flatten, uniform
                    new_img[new_i][new_j] = img[i][j]
                for j,new_j in enumerate(range(int(right_br_x-bl_r), int(right_br_x+bl_r))):  # linear decreasing
                    new_img[new_i][new_j] += (((2*bl_r-j)/float(2*bl_r)) * img[i][int(j + right_br_x - bl_r)]).astype(np.uint8)

            elif img_num == image_set_size-1:
                right_br_x = new_w
                for j,new_j in enumerate(range(int(left_br_x-bl_r), int(left_br_x+bl_r))): # linear increasing
                    new_img[new_i][new_j] += ((j/float(2*bl_r)) * img[i][int(j + (left_br_x-bl_r) - shift_acc[img_num][0])]).astype(np.uint8)
                for j,new_j in enumerate(range(int(left_br_x+bl_r), new_w)): # flatten, uniform
                    new_img[new_i][new_j] = img[i][ int(j + (left_br_x+bl_r) - shift_acc[img_num][0])]

            else:
                right_br_x = ((shift_acc[img_num][0]+width) + shift_acc[img_num+1][0])/2
                for j,new_j in enumerate(range(int(left_br_x-bl_r), int(left_br_x+bl_r))): # linear increasing
                    new_img[new_i][new_j] += ((j/float(2*bl_r)) * img[i][int(j + (left_br_x-bl_r) - shift_acc[img_num][0])]).astype(np.uint8)
                for j,new_j in enumerate(range(int(left_br_x+bl_r), int(right_br_x-bl_r))): # flatten, uniform
                    new_img[new_i][new_j] = img[i][ int(j + (left_br_x+bl_r) - shift_acc[img_num][0])]
                for j,new_j in enumerate(range(int(right_br_x-bl_r), int(right_br_x+bl_r))):  # linear decreasing
                    new_img[new_i][new_j] += (((2*bl_r-j)/float(2*bl_r)) * img[i][int(j + (right_br_x-bl_r) - shift_acc[img_num][0])]).astype(np.uint8)
        left_br_x = right_br_x
    return new_img

def feature_matching(imagename, kpt, dt, kpi, di):
    img = cv2.imread(imagename)
    height, width =  img.shape[:2]
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(np.asarray(di, np.float32), flann_params)
    idx, dist = flann.knnSearch(np.asarray(dt, np.float32), 1, params={})
    del flann
    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1).tolist()
    idx = idx.reshape(-1).tolist()
    indices = list(range(len(dist)))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]

    kpi_cut = []
    for i, dis in zip(idx, dist):
        if dis < args.cutoff:
            kpi_cut.append(kpi[i])
        else:
            break

    kpt_cut = []
    for i, dis in zip(indices, dist):
        if dis < args.cutoff:
            kpt_cut.append(kpt[i])
        else:
            break

    matched_pairs = []
    matched_x_max_thres = width - width/args.x_thres_partition
    matched_x_min_thres = width / args.x_thres_partition
    matched_y_abs_thres = height / args.y_thres_partition

    for i in range(np.array(kpi_cut).shape[0]):
        distance_x = kpt_cut[i][1] - kpi_cut[i][1]
        distance_y = abs(kpt_cut[i][0] - kpi_cut[i][0])
        if distance_y<matched_y_abs_thres and distance_x < matched_x_max_thres and distance_x > matched_x_min_thres:
            pt_a = (int(kpt_cut[i][1]), int(kpt_cut[i][0]))
            pt_b = (int(kpi_cut[i][1]), int(kpi_cut[i][0]))
            matched_pairs.append([pt_a,pt_b])
    return matched_pairs

def feature_detector(imagename):
    s = 3
    k = 2 ** (1.0 / s)
    original = Image.open(imagename).convert('L')
    k1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * np.power(k , 2), 1.6 * np.power(k, 3), 1.6 * np.power(k, 4)])
    k2 = np.array([1.6 * np.power(k, 2), 1.6 * np.power(k, 3), 1.6 * np.power(k, 4), 1.6 * np.power(k, 5),
                      1.6 * np.power(k, 6), 1.6 * np.power(k, 7)])
    k3 = np.array([1.6 * np.power(k, 5), 1.6 * np.power(k, 6), 1.6 * np.power(k, 7), 1.6 * np.power(k, 8),
                      1.6 * np.power(k, 9), 1.6 * np.power(k, 10)])
    k4 = np.array([1.6 * np.power(k, 8), 1.6 * np.power(k, 9), 1.6 * np.power(k, 10), 1.6 * np.power(k, 11),
                      1.6 * (k ** 12), 1.6 * (k ** 13)])
    ktotal = np.array([1.6, 1.6 * k, 1.6 * np.power(k, 2), 1.6 * np.power(k, 3), 1.6 * np.power(k, 4),
                          1.6 * np.power(k, 5), 1.6 * np.power(k, 6), 1.6 * np.power(k, 7), 1.6 * np.power(k, 8),
                          1.6 * np.power(k, 9), 1.6 * np.power(k, 10), 1.6 * np.power(k, 11)])

    doubled = original.resize((original.size[0] * 2, original.size[1] * 2), Image.BILINEAR)
    doubled = np.array(doubled).astype(int)
    normal = original.resize(original.size, Image.BILINEAR)
    normal = np.array(normal).astype(int)
    halved = original.resize((original.size[0] // 2, original.size[1] // 2), Image.BILINEAR)
    halved = np.array(halved).astype(int)
    quartered = original.resize((original.size[0] // 4, original.size[1] // 4), Image.BILINEAR)
    quartered = np.array(quartered).astype(int)

    # Initialize Gaussian pyramids
    py1 = np.zeros((doubled.shape[0], doubled.shape[1], 6))
    py2 = np.zeros((normal.shape[0], normal.shape[1], 6))
    py3 = np.zeros((halved.shape[0], halved.shape[1], 6))
    py4 = np.zeros((quartered.shape[0], quartered.shape[1], 6))

    # Construct Gaussian pyramids
    img = Image.fromarray(np.uint8(doubled)).convert("L")
    for i in range(0, 6):
        blurred = img.filter(ImageFilter.GaussianBlur(radius=k1[i]))
        py1[:,:,i] = np.array(blurred)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=k2[i]))
        resized = blurred.resize((int(img.width * 0.5), int(img.height * 0.5)), Image.BILINEAR)
        py2[:,:,i] = np.array(resized)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=k3[i]))
        resized = blurred.resize((int(img.width * 0.25), int(img.height * 0.25)), Image.BILINEAR)
        py3[:, :, i] = np.array(resized)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=k4[i]))
        resized = blurred.resize((int(img.width * 0.125), int(img.height * 0.125)), Image.BILINEAR)
        py4[:,:,i] = np.array(resized)

    # Initialize Difference-of-Gaussians (DoG) pyramids
    dpy1 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    dpy2 = np.zeros((normal.shape[0], normal.shape[1], 5))
    dpy3 = np.zeros((halved.shape[0], halved.shape[1], 5))
    dpy4 = np.zeros((quartered.shape[0], quartered.shape[1], 5))

    # Construct DoG pyramids
    for i in range(0, 5):
        dpy1[:,:,i] = py1[:,:,i+1] - py1[:,:,i]
        dpy2[:,:,i] = py2[:,:,i+1] - py2[:,:,i]
        dpy3[:,:,i] = py3[:, :, i + 1] - py3[:, :, i]
        dpy4[:,:,i] = py4[:,:,i+1] - py4[:,:,i]

    # Initialize pyramids to store extrema locations
    extrpy1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    extrpy2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    extrpy3 = np.zeros((halved.shape[0], halved.shape[1], 3))
    extrpy4 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    iter_params = [80,40,20,10]
    dpys = [dpy1,dpy2,dpy3,dpy4]
    extrpys = [extrpy1,extrpy2,extrpy3,extrpy4]
    shapes = [doubled.shape, normal.shape, halved.shape, quartered.shape]

    for z,diffpy in enumerate(dpys):
        sys.stdout.write(" | | "+str(z)+" octave...  ");sys.stdout.flush()
        for i in range(1, 4):
            for j in range(iter_params[z], shapes[z][0] - iter_params[z]):
                for k in range(iter_params[z], shapes[z][1] - iter_params[z]):
                    temp = diffpy[j, k, i]
                    if np.absolute(temp) < args.threshold:
                        continue
                    maxbool = (temp > 0)
                    minbool = (temp < 0)

                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            for dk in range(-1, 2):
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                maxbool = maxbool and (temp > diffpy[j + dj, k + dk, i + di])
                                minbool = minbool and (temp < diffpy[j + dj, k + dk, i + di])
                                if not maxbool and not minbool:
                                    break
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break

                    if maxbool or minbool:
                        a = diffpy[j+1, k, i]
                        b = diffpy[j, k+1, i]
                        c = diffpy[j, k-1, i]
                        d = diffpy[j, k, i+1]
                        e = diffpy[j-1, k, i]
                        f = diffpy[j, k, i-1]

                        dx = (b - c) * 0.5 / 255
                        dy = (a - e) * 0.5 / 255
                        ds = (d - f) * 0.5 / 255
                        dxx = (b + c - 2 * temp) * 1.0 / 255
                        dyy = (a + e - 2 * temp) * 1.0 / 255
                        dss = (d + f - 2 * temp) * 1.0 / 255
                        dxy = (diffpy[j+1, k+1, i] - diffpy[j+1, k-1, i] - diffpy[j-1, k+1, i] + diffpy[j-1, k-1, i]) * 0.25 / 255
                        dxs = (diffpy[j, k+1, i+1] - diffpy[j, k-1, i+1] - diffpy[j, k+1, i-1] + diffpy[j, k-1, i-1]) * 0.25 / 255
                        dys = (diffpy[j+1, k, i+1] - diffpy[j-1, k, i+1] - diffpy[j+1, k, i-1] + diffpy[j-1, k, i-1]) * 0.25 / 255

                        dD = np.matrix([[dx], [dy], [ds]])
                        H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                        x_hat = numpy.linalg.lstsq(H, dD,rcond=None)[0]
                        D_x_hat = temp + 0.5 * np.dot(dD.transpose(), x_hat)

                        r = 10.0
                        if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):
                            extrpys[z][j, k, i - 1] = 1

    # Gradient magnitude and orientation for each image sample point at each scale
    magpyr1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    magpyr2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    magpyr3 = np.zeros((halved.shape[0], halved.shape[1], 3))
    magpyr4 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    o1 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    o2 = np.zeros((normal.shape[0], normal.shape[1], 3))
    o3 = np.zeros((halved.shape[0], halved.shape[1], 3))
    o4 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    for i in range(0, 3):
        for j in range(1, doubled.shape[0] - 1):
            for k in range(1, doubled.shape[1] - 1):
                magpyr1[j, k, i] = np.sqrt(np.square((doubled[j+1, k] - doubled[j-1, k])) + np.square((doubled[j, k+1] - doubled[j, k-1])))
                o1[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((doubled[j, k+1] - doubled[j, k-1]), (doubled[j+1, k] - doubled[j-1, k])))
        for j in range(1, normal.shape[0] - 1):
            for k in range(1, normal.shape[1] - 1):
                magpyr2[j, k, i] = np.sqrt(np.square((normal[j+1, k] - normal[j-1, k])) + np.square((normal[j, k+1] - normal[j, k-1])))
                o2[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((normal[j, k+1] - normal[j, k-1]), (normal[j+1, k] - normal[j-1, k])))
        for j in range(1, halved.shape[0] - 1):
            for k in range(1, halved.shape[1] - 1):
                magpyr3[j, k, i] = np.sqrt(np.square((halved[j+1, k] - halved[j-1, k])) + np.square((halved[j, k+1] - halved[j, k-1])))
                o3[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((halved[j, k+1] - halved[j, k-1]), (halved[j+1, k] - halved[j-1, k])))
        for j in range(1, quartered.shape[0] - 1):
            for k in range(1, quartered.shape[1] - 1):
                magpyr4[j, k, i] = np.sqrt(np.square((quartered[j+1, k] - quartered[j-1, k])) + np.square((quartered[j, k+1] - quartered[j, k-1])))
                o4[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((quartered[j, k+1] - quartered[j, k-1]), (quartered[j+1, k] - quartered[j-1, k])))

    extr_sum = np.sum(extrpy1) + np.sum(extrpy2) + np.sum(extrpy3) + np.sum(extrpy4)
    keypoints = np.zeros((int(extr_sum), 4))

    print(" Calculating keypoint orientations...\n")

    count = 0

    for i in range(0, 3):
        for j in range(80, doubled.shape[0] - 80):
            for k in range(80, doubled.shape[1] - 80):
                if extrpy1[j, k, i] == 1:
                    gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * ktotal[i]) ** 2))
                    two_sd = np.floor(2 * 1.5 * ktotal[i])
                    orient_hist = np.zeros([36,1])
                    for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
                        ylim = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > doubled.shape[0] - 1 or k + y < 0 or k + y > doubled.shape[1] - 1:
                                continue
                            weight = magpyr1[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
                            bin_idx = np.clip(np.floor(o1[j + x, k + y, i]), 0, 35)
                            bin_idx = int(np.floor(bin_idx))
                            orient_hist[bin_idx] += weight

                    maxv = np.amax(orient_hist)
                    maxidx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), ktotal[i], maxidx])
                    count += 1
                    orient_hist[maxidx] = 0
                    newmaxval = np.amax(orient_hist)
                    while newmaxval > 0.8 * maxv:
                        newmaxidx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), ktotal[i], newmaxidx]]), axis=0)
                        orient_hist[newmaxidx] = 0
                        newmaxval = np.amax(orient_hist)
        for j in range(40, normal.shape[0] - 40):
            for k in range(40, normal.shape[1] - 40):
                if extrpy2[j, k, i] == 1:
                    gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * ktotal[i + 3]) ** 2))
                    two_sd = np.floor(2 * 1.5 * ktotal[i + 3])
                    orient_hist = np.zeros([36,1])
                    for x in range(int(-1 * two_sd), int(two_sd + 1)):
                        ylim = int(np.sqrt(np.square(two_sd) - np.square(np.absolute(x))))
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > normal.shape[0] - 1 or k + y < 0 or k + y > normal.shape[1] - 1:
                                continue
                            weight = magpyr2[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
                            bin_idx = np.clip(np.floor(o2[j + x, k + y, i]), 0, 35)
                            bin_idx = int(np.floor(bin_idx))
                            orient_hist[bin_idx] += weight
                    maxv = np.amax(orient_hist)
                    maxidx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j, k, ktotal[i + 3], maxidx])
                    count += 1
                    orient_hist[maxidx] = 0
                    newmaxval = np.amax(orient_hist)
                    while newmaxval > 0.8 * maxv:
                        newmaxidx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([[j, k, ktotal[i + 3], newmaxidx]]), axis=0)
                        orient_hist[newmaxidx] = 0
                        newmaxval = np.amax(orient_hist)
        for j in range(20, halved.shape[0] - 20):
            for k in range(20, halved.shape[1] - 20):
                if extrpy3[j, k, i] == 1:
                    gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * ktotal[i + 6]) ** 2))
                    two_sd = np.floor(2 * 1.5 * ktotal[i + 6])
                    orient_hist = np.zeros([36,1])
                    for x in range(int(-1 * two_sd * 0.5), int(two_sd * 0.5) + 1):
                        ylim = int(np.sqrt(np.square((two_sd * 0.5)) - np.square(np.absolute(x))))
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > halved.shape[0] - 1 or k + y < 0 or k + y > halved.shape[1] - 1:
                                continue
                            weight = magpyr3[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
                            bin_idx = np.clip(np.floor(o3[j + x, k + y, i]), 0, 35)
                            bin_idx = int(np.floor(bin_idx))
                            orient_hist[bin_idx] += weight

                    maxv = np.amax(orient_hist)
                    maxidx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j * 2, k * 2, ktotal[i + 6], maxidx])
                    count += 1
                    orient_hist[maxidx] = 0
                    newmaxval = np.amax(orient_hist)
                    while newmaxval > 0.8 * maxv:
                        newmaxidx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([[j * 2, k * 2, ktotal[i + 6], newmaxidx]]), axis=0)
                        orient_hist[newmaxidx] = 0
                        newmaxval = np.amax(orient_hist)
        for j in range(10, quartered.shape[0] - 10):
            for k in range(10, quartered.shape[1] - 10):
                if extrpy4[j, k, i] == 1:
                    gaussian_window = multivariate_normal(mean=[j, k], cov=((1.5 * ktotal[i + 9]) ** 2))
                    two_sd = np.floor(2 * 1.5 * ktotal[i + 9])
                    orient_hist = np.zeros([36,1])
                    for x in range(int(-1 * two_sd * 0.25), int(two_sd * 0.25) + 1):
                        ylim = int(np.sqrt(np.square(two_sd * 0.25) - np.square(np.absolute(x))))
                        for y in range(-1 * ylim, ylim + 1):
                            if j + x < 0 or j + x > quartered.shape[0] - 1 or k + y < 0 or k + y > quartered.shape[1] - 1:
                                continue
                            weight = magpyr4[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
                            bin_idx = np.clip(np.floor(o4[j + x, k + y, i]), 0, 35)
                            bin_idx = int(np.floor(bin_idx))
                            orient_hist[bin_idx] += weight

                    maxv = np.amax(orient_hist)
                    maxidx = np.argmax(orient_hist)
                    keypoints[count, :] = np.array([j * 4, k * 4, ktotal[i + 9], maxidx])
                    count += 1
                    orient_hist[maxidx] = 0
                    newmaxval = np.amax(orient_hist)
                    while newmaxval > 0.8 * maxv:
                        newmaxidx = np.argmax(orient_hist)
                        np.append(keypoints, np.array([[j * 4, k * 4, ktotal[i + 9], newmaxidx]]), axis=0)
                        orient_hist[newmaxidx] = 0
                        newmaxval = np.amax(orient_hist)

    print("Calculating descriptor...\n")

    magpyr = np.zeros((normal.shape[0], normal.shape[1], 12))
    opyr = np.zeros((normal.shape[0], normal.shape[1], 12))

    for i in range(0, 3):  # Looping through the first 3 layers
        img_mag = Image.fromarray(np.uint8(magpyr1[:, :, i]))
        resized_img_mag = img_mag.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        magpyr[:, :, i] = np.array(resized_img_mag, dtype=float)

        magmax = np.amax(magpyr1[:, :, i])
        magpyr[:, :, i] *= (magmax / np.amax(magpyr[:, :, i]))

        img_ori = Image.fromarray(np.uint8(o1[:, :, i]))
        resized_img_ori = img_ori.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        opyr[:, :, i] = np.array(resized_img_ori, dtype=int)

        opyr[:, :, i] = ((36.0 / np.amax(opyr[:, :, i])) * opyr[:, :, i]).astype(int)

    for i in range(0, 3):
        magpyr[:, :, i+3] = (magpyr2[:, :, i]).astype(float)
        opyr[:, :, i+3] = (o2[:, :, i]).astype(int)

    for i in range(0, 3):
        img_mag = Image.fromarray(np.uint8(magpyr3[:, :, i]))
        resized_img_mag = img_mag.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        magpyr[:, :, i+6] = np.array(resized_img_mag, dtype=int)

        img_ori = Image.fromarray(np.uint8(o3[:, :, i]))
        resized_img_ori = img_ori.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        opyr[:, :, i+6] = np.array(resized_img_ori, dtype=int)

    for i in range(0, 3):
        img_mag = Image.fromarray(np.uint8(magpyr4[:, :, i]))
        resized_img_mag = img_mag.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        magpyr[:, :, i+9] = np.array(resized_img_mag, dtype=int)

        img_ori = Image.fromarray(o4[:, :, i])
        resized_img_ori = img_ori.resize((normal.shape[1], normal.shape[0]), Image.BILINEAR)
        opyr[:, :, i+9] = np.array(resized_img_ori, dtype=int)


    descriptors = np.zeros([keypoints.shape[0], 128])

    for i in range(0, keypoints.shape[0]):
        for x in range(-8, 8):
            for y in range(-8, 8):
                theta = 10 * keypoints[i,3] * np.pi / 180.0
                xrot = int(np.round((np.cos(theta) * x) - (np.sin(theta) * y)))
                yrot = int(np.round((np.sin(theta) * x) + (np.cos(theta) * y)))
                scale_idx = int(np.argwhere(ktotal == keypoints[i,2])[0][0])
                x0 = int(keypoints[i,0])
                y0 = int(keypoints[i,1])
                gaussian_window = multivariate_normal(mean=[x0,y0], cov=8)
                weight = magpyr[x0 + xrot, y0 + yrot, scale_idx] * gaussian_window.pdf([x0 + xrot, y0 + yrot])
                angle = opyr[x0 + xrot, y0 + yrot, scale_idx] - keypoints[i,3]
                if angle < 0:
                    angle = 36 + angle

                bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
                descriptors[i, 32 * int((x + 8)/4) + 8 * int((y + 8)/4) + bin_idx] += weight

        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
        descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])

    newimg = cv2.imread(imagename).copy()
    for kp in keypoints:
        pt_a = (int(kp[1]), int(kp[0]))
        cv2.circle(newimg, pt_a, 3, (147,20,255), -1)
    cv2.imwrite(imagename[:-4]+'_features.jpg', newimg)

    return [keypoints, descriptors]

# stiching from left to right
if __name__ == '__main__':
    # read files
    image_dir = args.path
    image_set, focal_length = load_images_and_focal_lengths(image_dir)
    image_set_size, height, width, _ = image_set.shape
    print('image_set',image_set.shape, 'focal_length set',focal_length.shape)

    # project to cylinder coordinate
    cylinder_projs = cylindrical_projection(image_set, focal_length)
    fig1=plt.figure().suptitle('cylindrical_projection')
    for i,cylinder_proj in enumerate(cylinder_projs):
        cylinder_proj_rgb = cylinder_proj[:,:,::-1]
        subfig = plt.subplot(2,math.ceil(image_set_size/2.),i+1)
        subfig.imshow(cylinder_proj_rgb)
        cv2.imwrite(str(i)+".jpg",cylinder_proj)

    # initail values
    stitched_image = cylinder_projs[0].copy()
    shifts = [[0,0]]
    feature_cache = [[], []]
    for i in range(1, image_set_size):
        print('Computing ...... {}/{}'.format(str(i),str(image_set_size-1)))
        sys.stdout.flush()
        img1 = str(i-1)+".jpg"
        img2 = str(i)+".jpg"
        img1_cv = cylinder_projs[i-1].copy()
        img2_cv = cylinder_projs[i].copy()

        keypints1, descriptors1 = feature_cache
        if i==1: #first loop
            keypints1, descriptors1 = feature_detector(img1)
        keypints2, descriptors2 = feature_detector(img2)
        feature_cache = [keypints2, descriptors2]

        print('Feature matching .... ')
        matched_pairs = feature_matching(img2, keypints1, descriptors1, keypints2, descriptors2)

        print('Find best shift using RANSAC .... ')
        shift = RANSAC(matched_pairs)
        shifts.append(shift)
        print('best shift ', shift)

    # stitching and blending
    print('Stitching image ... ', flush=True)
    stitched_image = image_matching(shifts, image_set_size, height, width)
    cv2.imwrite('pano.png', stitched_image)
    pano_image = stitched_image.copy()

    if args.ALIGN:
        print('End to end alignment ... ', flush=True)
        aligned_image = alingment(pano_image, shifts)
        cv2.imwrite('aligned.png', aligned_image)
        pano_image = aligned_image.copy()

    if args.CROP:
        print('Rectangling image ... ', flush=True)
        cropped_image = rectangling(pano_image)
        cv2.imwrite('result.png', cropped_image)
        pano_image = cropped_image.copy()

