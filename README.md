## Rectifying Non-Stereo Images to Make Anaglyphs

This repository uses two images of a scene to create an anaglyph.  Instead of using images taken with a stereo set up, the two images can be taken with a handheld camera.  The images are then rectified to create an anaglyph that appears 3D when viewed with red/cyan glasses.

The results below use Scale Invariant Feature Transform (SIFT) for feature detection combined with Fast Library for Approximate Nearest Neighbors (FLANN) for feature matching.

This project was completed as a Computational Photography assignment at Georgia Institute of Technology.

### Results



![Motorcycle anaglyph](images/final_cropped_images/cropped_motorbike.jpg "Motorcycle anaglyph") | ![Cyclops anaglyph](images/final_cropped_images/cropped_cyclops.jpg "Cyclops anaglyph") | ![Sports oval anaglyph](images/final_cropped_images/cropped_oval.jpg "Sports oval anaglyph")

### Processing Pipeline
![Image processing Pipeline](images/md/image_processing_pipeline.png "Image processing pipeline")
Motorbike example
![Motorbike pipeline](images/md/motorbike_processing.png "Motorbike image processsing")




### Motorbikes

Left photo            |  Right photo
:-------------------------:|:-------------------------:
![Motorcycle Image 1](images/source/motorbike/image1_small.jpg "Motorcycle picture 1") | ![Motorcycle Image 2](images/source/motorbike/image2_small.jpg "Motorcycle picture 2") 
:-------------------------:|:-------------------------:
Computed anaglyph            |  After rotation and cropping
:-------------------------:|:-------------------------:
![Motorcycle new_anaglyph](images/output/motorbike/anaglyph_new.jpg "Motorcycle anaglyph") | ![Motorcycle anaglyph](images/final_cropped_images/cropped_motorbike.jpg "Motorcycle anaglyph")


### Cyclops statue
Left photo
![Cyclops Image 1](images/source/cyclops/image1_small.jpg "Cyclops picture 1")
Right photograph
![Cyclops Image 2](images/source/cyclops/image2_small.jpg "Cyclops picture 2")
Anaglyph
![Cyclops new_anaglyph](images/output/cyclops/anaglyph_new.jpg "Cyclops anaglyph")
Final result after rotation and cropping
![Cyclops anaglyph](images/final_cropped_images/cropped_cyclops.jpg "Cyclops anaglyph")

### Sports Oval
Left photo
![Sports oval Image 1](images/source/oval/image1_small.jpg "Oval picture 1")
Right photo
![Sports oval Image 2](images/source/oval/image2_small.jpg "Oval picture 2")
Anaglyph
![Sports oval new_anaglyph](images/output/oval/anaglyph_new.jpg "Oval anaglyph")
Final result after rotation and cropping
![Cyclops anaglyph](images/final_cropped_images/cropped_oval.jpg "Oval anaglyph")

### References

Blue Lightning TV Photoshop, 2013, Photoshop Tutorial: How to Make Jaw-dropping, 3-D Anaglyphs from Photos, online video, accessed 27 Nov 2017, https://www.youtube.com/watch?v=Mh5qiCvaS0o

Hartley, R. and Zisserman, A., 2003. Multiple view geometry in computer vision. Cambridge University Press.

Loop, C. and Zhang, Z., 1999. Computing rectifying homographies for stereo vision. In Computer Vision and Pattern Recognition, 1999. IEEE Computer Society Conference on. (Vol. 1, pp. 125-131). IEEE.

Mathworks, Uncalibrated Stereo Image Rectification https://au.mathworks.com/help/vision/examples/uncalibrated-stereo-image-rectification.html

Middlebury Stereo Datasets http://vision.middlebury.edu/stereo/data/
Open CV 2.4 Documentation https://docs.opencv.org/2.4/

Open CV, Epipolar Geometry
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

Solem, J.E., 2012. Programming Computer Vision with Python: Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".

Wikipedia, Image Rectification, https://en.wikipedia.org/wiki/Image_rectification  Accessed 3 December 2017

https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system

https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

https://stackoverflow.com/questions/41760798/opencv-python-stereo-match-py