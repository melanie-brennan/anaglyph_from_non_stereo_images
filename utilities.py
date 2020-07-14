import cv2
import numpy as np

#return a grey image
def convert_to_grey_scale(colour_img):
    """Converts an OpenCV BGR image to greyscale
    :param colour_img:
    :return: grey scale image
    """
    return cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)


def get_image_corners(image):
    """ Code from previous assignment (supplied)
    Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)

    # get the height and width of the image
    height, width = image.shape[:2]

    # get the coordinates of the corners.
    corners[0] = (0, 0)  # bottom left
    corners[1] = (0, height)  # top left
    corners[2] = (width, height)  # top right
    corners[3] = (width, 0)  # bottom right

    return corners


def get_bounding_corners(corners_1, corners_2, homography1, homography2):
    """ Code adapted from previous assignment (supplied)
    Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography1 : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)
    """

    # Use the homography to transform the perspective of the corners
    transformed_corners_1 = cv2.perspectiveTransform(corners_1, homography1)
    transformed_corners_2 = cv2.perspectiveTransform(corners_2, homography2)

    # get the minimum and maximum x and y values
    # note that ymin etc are results of min and max funtions so they are arrays
    # the min/max of the transformed image 1 are also compare with image 2
    #ymin = min(min(transformed_corners_1[:, :, 1]), min(transformed_corners_2[:, :, 1]))
    #ymax = max(max(transformed_corners_1[:, :, 1]), max(transformed_corners_2[:, :, 1]))
    #xmin = min(min(transformed_corners_1[:, :, 0]), min(transformed_corners_2[:, :, 0]))
    #xmax = max(max(transformed_corners_1[:, :, 0]), max(transformed_corners_2[:, :, 0]))

    #cropped version
    ymin = max(min(transformed_corners_1[:, :, 1]), min(transformed_corners_2[:, :, 1]))
    ymax = min(max(transformed_corners_1[:, :, 1]), max(transformed_corners_2[:, :, 1]))
    xmin = max(min(transformed_corners_1[:, :, 0]), min(transformed_corners_2[:, :, 0]))
    xmax = min(max(transformed_corners_1[:, :, 0]), max(transformed_corners_2[:, :, 0]))

    # put the x and y values in tuples to represent corners
    top_left = np.array([xmin[0], ymin[0]], dtype=np.float64)
    bottom_right = np.array([xmax[0], ymax[0]], dtype=np.float64)

    return top_left, bottom_right


def wrap_canvas(image, homography, min_xy, max_xy):
    #Code from previous assignment (supplied)
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)
    """

    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE
    x_min, y_min = min_xy

    #Create the translate matrix
    M_translate = np.asarray([[1, 0, -x_min],[0, 1, -y_min],[0, 0, 1]], dtype = np.int)

    #calculate the matrix for homography and translation using dot product (note order matters!)
    M_trans_hom = M_translate.dot(homography)

    #perform warp
    img_warped = cv2.warpPerspective(image, M_trans_hom, canvas_size)

    return img_warped


def detect_features_with_sift(img1, img2):
    """
    # Detects image features using OpenCV SIFT function (Scale Invariant Feature Transform)
    :param img1: left image
    :param img2: right image
    :return: keypoints and descriptors
    """

    # Obtainment of the correspondent point with SIFT
    #featureDetector = cv2.SIFT()  #Open CV2
    featureDetector = cv2.xfeatures2d.SIFT_create()  #Open CV3 and Open CV4

    ###find the keypoints and descriptors with SIFT
    kp1, des1 = featureDetector.detectAndCompute(img1, None)
    kp2, des2 = featureDetector.detectAndCompute(img2, None)

    return kp1, kp2, des1, des2


def detect_features_with_orb(image_1, image_2):
    #Function adapeted from the panorama assignment
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    """
    #feature_detector = cv2.ORB(nfeatures=10000)   # Open CV2
    feature_detector = cv2.ORB_create(nfeatures=10000) #Open CV3 and Open CV4

    image_1_kp, image_1_desc = feature_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feature_detector.detectAndCompute(image_2, None)

    return image_1_kp, image_2_kp, image_1_desc, image_2_desc


def match_features_with_brute_force(keypoints1, keypoints2, descriptors1, descriptors2, num_matches):
    """
    Uses a brute force method to match features
    :param keypoints1: keypoints from the left image
    :param keypoints2: keypints from the right image
    :param descriptors1: descriptors from the left image
    :param descriptors2: descriptors from the right image
    :param num_matches: the number of top matches to be used
    :return: matching points (points1 and points2) and the matches
    """

    #Adapted from previous assignment
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(descriptors1, descriptors2), key=lambda x: x.distance)[:num_matches]
    points1 = []
    points2 = []

    for i in range(len(matches)):
        img1_idx = matches[i].queryIdx
        img2_idx = matches[i].trainIdx

        points1.append(keypoints1[img1_idx].pt)
        points2.append(keypoints2[img2_idx].pt)

    return points1, points2, matches


def match_features_with_flann(keypoints1, keypoints2, descriptors1, descriptors2):

    """ Match features using Fast Library for Approximate Nearest Neighbors
    :param keypoints1:
    :param keypoints2:
    :param descriptors1:
    :param descriptors2:
    :return: The points (points1 and points2) and good_matches which are within a 0.8 distance ratio
    Note: See https: // stackoverflow.com / questions / 36172913 / opencv - depth - map -from-uncalibrated - stereo - system
    """

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    points1 = []
    points2 = []

    ###ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            points2.append(keypoints2[m.trainIdx].pt)
            points1.append(keypoints1[m.queryIdx].pt)

    return points1, points2, good_matches


def make_anaglyph(image_left, image_right):
    """Create a red/cyan anaglyph image using grayscale left and right images.
    Image is in BGR format.  Right image goes in the blue and green channels, left image in the red channel
    :param image_left: numpy.ndarray representing left image
    :param image_right:  numpy.ndarray representing right image
    :return:  numpy.ndarray representing a 3 channel image
    """

    return np.dstack([image_right, image_right, image_left])