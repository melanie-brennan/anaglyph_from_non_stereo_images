import numpy as np
import cv2

import utilities as util


def process_two_images(image1, image2, featureMatcherMethod = "SIFTFLANN"):
    """
    Processes two images to produce intermediate images and the final anaglyph
    :param image1: left photo
    :param image2: right photo
    :param featureMatcherMethod: Combination of feature detetector and feature matching method.
    Default is a combination of SIFT feature detector and FLANN matchind
    :return: intermediate images and the final anaglyph (image1_small, image2_small, image1_grey, image2_grey,
    image_warped_1, image_warped_2, image_anaglyph_original, image_anaglyph_new)
    """

    # RESIZE IMAGES AUTOMATICALLY
    image1_small = cv2.resize(image1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    image2_small = cv2.resize(image2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # CONVERT IMAGES TO GREYSCALE
    #convert images to greyscale
    image1_grey = util.convert_to_grey_scale(image1_small)
    image2_grey = util.convert_to_grey_scale(image2_small)

    # FIND FEATURES IN THE IMAGES
    # Feature Detection is SIFT and FLANN  matching (SIFTFLANN) (default)
    # or ORB with brute force matching

    if featureMatcherMethod == "SIFTFLANN":
        keypoints1, keypoints2, descriptors1, descriptors2 = util.detect_features_with_sift(image1_grey, image2_grey)
    elif featureMatcherMethod == "ORBBRUTE":
        keypoints1, keypoints2, descriptors1, descriptors2 = util.detect_features_with_orb(image1_grey, image2_grey)
    else:
        print("Invalid feature matcher")
        exit()

    # MATCH FEATURES
    if featureMatcherMethod == "SIFTFLANN":
        points1, points2, good_matches = util.match_features_with_flann(keypoints1, keypoints2, descriptors1, descriptors2)

    elif featureMatcherMethod == "ORBBRUTE":
        points1, points2, good_matches = util.match_features_with_brute_force(keypoints1, keypoints2, descriptors1, descriptors2, 10000)

    # FIND THE FUNDAMENTAL MATRIX
    #Convert the points to an array
    points1 = np.array(points1)
    points2 = np.array(points2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

    # PERFORM RECTIFICATION
    #Flatten the mask and use it to filter points
    #https: // stackoverflow.com / questions / 36172913 / opencv - depth - map -from-uncalibrated - stereo - system
    points1 = points1[:, :][mask.ravel() == 1]
    points2 = points2[:, :][mask.ravel() == 1]

    # Convert the points to ints and reshape into a long Nx1 matrix
    points1 = np.int32(points1)
    points2 = np.int32(points2)

    points1New = points1.reshape((points1.shape[0] * 2, 1))
    points2New = points2.reshape((points2.shape[0] * 2, 1))

    retVal, homography1, homography2 = cv2.stereoRectifyUncalibrated(points1New, points2New, F, image1_grey.shape)

    # GET CANVAS SIZE
    # get the corners of the original input images
    corners1 = util.get_image_corners(image1_grey)
    corners2 = util.get_image_corners(image2_grey)
    top_left, bottom_right = util.get_bounding_corners(corners1, corners2, homography1, homography2)

    # WARP THE IMAGES
    image_warped_1 = util.wrap_canvas(image1_grey, homography1, top_left, bottom_right)
    image_warped_2 = util.wrap_canvas(image2_grey, homography2, top_left, bottom_right)

    # MAKE THE ANAGLYPHS
    #make anaglyphs
    image_anaglyph_original = util.make_anaglyph(image1_grey, image2_grey)
    image_anaglyph_new = util.make_anaglyph(image_warped_1, image_warped_2)

    # RETURN
    return image1_small, image2_small, image1_grey, image2_grey, image_warped_1, image_warped_2, image_anaglyph_original, image_anaglyph_new