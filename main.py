import os
import errno
from os import path
import cv2   #Note,  OpenCV 3 and OpenCV 4 require the opencv-contrib-python be installed
import processImages as proc

NUM_MATCHES = 10
SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"
IMG_EXTS = set(["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp", "ppm"])


def main(image_files, output_folder):
    """ Reads images, calls image processing function and writes output images to file.
    :param image_files: left and right images
    :param output_folder: name of output folder that will contain resulting images
    :return: None
    """

    inputs = ((name, cv2.imread(name)) for name in sorted(image_files)
              if path.splitext(name)[-1][1:].lower() in IMG_EXTS)

    # start with the first image in the folder and process each image in order
    #name, my_image = inputs.next()  # Python 2
    name, my_image = next(inputs)  # Python 3
    print("  Starting with: {}".format(name))
    for name, next_img in inputs:

        if next_img is None:
            print("\nUnable to proceed: {} failed to load.".format(name))
            return

        print("  Adding {}".format(name))

        #process the images to make an anaglyph
        image_small_1, image_small_2, image_grey_1, image_grey_2, image_warped_1, image_warped_2, \
        image_anaglyph_original, image_anaglyph_new = proc.process_two_images(my_image, next_img, "SIFTFLANN")

    #Write images to file
    #Image writes were commented out for submission
    cv2.imwrite(path.join(output_folder, "image1_small.jpg"), image_small_1)
    cv2.imwrite(path.join(output_folder, "image2_small.jpg"), image_small_2)
    cv2.imwrite(path.join(output_folder, "image2_grey.jpg"), image_grey_2)
    cv2.imwrite(path.join(output_folder, "image1_grey.jpg"), image_grey_1)
    cv2.imwrite(path.join(output_folder, "image2_grey.jpg"), image_grey_2)
    cv2.imwrite(path.join(output_folder, "image1_warped.jpg"), image_warped_1)
    cv2.imwrite(path.join(output_folder, "image2_warped.jpg"), image_warped_2)
    cv2.imwrite(path.join(output_folder, "anaglyph_original.jpg"), image_anaglyph_original)

    #The line below creates the final anglyph ("anaglyph_new.jpg")
    cv2.imwrite(path.join(output_folder, "anaglyph_new.jpg"), image_anaglyph_new)

    #Last step is to manually rotate and crop the images images.  See the "final_cropped_images_folder"


if __name__ == "__main__":

    """Generate from the images in each subdirectory of SRC_FOLDER"""
    subfolders = os.walk(SRC_FOLDER)
    #subfolders.next()  # skip the root input folder   (Python 2)
    next(subfolders)  # skip the root input folder     (Python 3)
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print("\nProcessing '" + image_dir + "' folder...")
        image_files = [os.path.join(dirpath, name) for name in fnames]
        main(image_files, output_dir)