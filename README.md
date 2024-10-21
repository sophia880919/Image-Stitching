# Image-Stitching-
Panoramic images are created by stitching multiple shots of the same scene, offering a broader view and more environmental details. This Python program allows users to convert a series of continuous images into a panoramic image.

## Implement 
Step1. Image and Focal Length Loading 
The corresponding implementations load_images_and_focal_lengths(source_dir). 
We capture 9 photos, and use AutoStitch to estimating the focal length, and save the values in info.txt. 

Step2. Cylindrical Projection 
The corresponding implementations is cylindrical_projection(imgs, focal_lengths). 
Before proceeding with feature point detection and stitching, we attempt to complete cylindrical 
projection (Cylindrical warping). The warping implemented here essentially applies the transformation 
method discussed in the lecture notes, converting each pixel from x, y to x', y'. 

![image](https://github.com/user-attachments/assets/64725aeb-62fb-4178-926d-4d6a3de547c4)

Step 3. Feature Detection 
The corresponding implementations is feature_detector(imagename). 
Scale-space extrema detection and Keypoint localization involve four octaves, corresponding to the 
original image scales of 2x, 1x, 0.5x, and 0.25x. Each scale undergoes six layers of Gaussian blurring 
to produce five layers of Difference of Gaussians (DoG). Each point is then compared with its 26 
neighboring points, and those that are larger or smaller are labeled as extrema. Subsequently, accurate 
keypoint localization is performed for these extremas. 
To calculate the orientation of each keypoint and then extract local features, pixels of size 16x16 are 
taken for each keypoint. These are then averaged and divided into 4x4 cells. Each cell computes 
gradient and orientation values, which are summarized into an 8-bin histogram. This results in a 
combined 128-dimensional data. Finally, by performing L2 normalization, a descriptor that represents 
the keypoint is obtained. 
When the threshold of feature detector higher, the number of detected keypoint will decrease, and the 
inference time also decrease. However, when the number of detected keypoint decrease, the 
performance was worse. 

![image](https://github.com/user-attachments/assets/96b02f32-747b-428f-8684-d84e2d7b9555)

Step 4. Feature Matching  
The corresponding implementations is feature_matching(imagename, kpt, dt, kpi, di). 
Feature matching is about finding the corresponding relationships between feature points in two 
images. We have implemented this by using kd-tree + knn-search and utilized the flann library. This 
approach identifies the smallest distance between feature vectors in different images as a matching 
relationship. 

Step 5. Image Matching 
The corresponding implementations is image_matching(shift_list, image_set_size, height, width). 
Here, we implemented RANSAC to find the optimal shift. Essentially, it involves randomly selecting 
a matched pair, calculating the shift, and then applying it back to the point coordinates of the 
subsequent photo in all matched pairs. This calculates the estimated point coordinates of the previous 
photo. The 2-norm distance is then computed between these estimated coordinates and the actual 
coordinates of the previous photo in the matched pair. If the distance is less than a threshold, it is 
considered an inlier. Finally, the shift with the most inliers is selected as the final shift. 

Step 6. End to end alignment & Step 7. Rectangling 
The corresponding implementations are alingment(img, shifts) and rectangling(img). 
For end-to-end alignment of panoramas that gradually slope upwards or downwards, the method used 
is quite simple, basically following the lecture notes to distribute the displacement across each column. 
For the rectangling part, first convert the image to grayscale, then if a row is determined to contain a 
proportion of no-information pixels (black, value=0) exceeding a certain threshold, the entire row is 
removed. This process is applied to trim the top and bottom, resulting in the final panoramic image.
