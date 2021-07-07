## Project: Advanced Lane Finding

### Goals:

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chess.png "Undistorted"
[image1_2]: ./output_images/undistorted.png "Undistorted"
[image2_1]: ./output_images/sobel_x.png "Sobel X"
[image2_2]: ./output_images/sobel_y.png "Sobel Y"
[image2_3]: ./output_images/gradient_magnitude.png "Magnitude"
[image2_4]: ./output_images/gradient_direction.png "Direction"
[image3]: ./output_images/perspective_transform.png "Perspective Transform"
[image4]: ./output_images/histogram2.png "Histogram"
[image5]: ./output_images/sliding_windows.png "Sliding Window"
[image6]: ./output_images/pipeline.png "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

For the image calibration the OpenCV offers the relevant functions like `findChessboardCorners` and `calibrateCamera`.
As input there are a number of images of a chessboard, taken from different angles with the same camera. The function `findChessboardCorners` is used to store the corners points in an array `image points` for each calibration image where the chessboard could be found. The `object points` will always be the same as the known coordinates of the chessboard with zero as 'z' coordinate because the chessboard is flat. The object points are then stored in an array called `objpoints`. The object and images points are fed to `calibrateCamera`, which return camera calibration and distortion coefficients. We use than the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera. 
The following result was obtained using the desribed procedure: 


![alt text][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To apply the distortion correction the function undistort were implemented: 
```python
def undistort(image):
    mtx, dist = calc_calibration_params()
    dst_img = cv2.undistort(image, mtx, dist, None, mtx)
    return dst_img
```
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1_2]

#### 2. Creation of binary image:

I used a combination of color and gradient thresholds to generate a binary image.
For each substep I've implemented a separate function: 

##### 2.1 Taking image gradients in x and y directions
The function `abs_sobel_thresh()` returns based on the input parameter either gradient in x or y direction and sets thresholds to identify pixels within a certain gradient range.
The following pictures demonstrates both methods: 

###### Gradient in x direction: Sobel X
![alt text][image2_1]
###### Gradient in x direction: Sobel Y
![alt text][image2_2]

##### 2.2 Taking magnitude of  gradient in both in x and y direction

The function `mag_thresh()` calculates the magnitude of the gradient, in both x and y and applies a threshold to this overall magnitude. The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients, which are returned by previously described function `abs_sobel_thresh()`. 
The below images shows the applied thresholded magnitude of derivatives: 

![alt text][image2_3]

##### 2.3 Taking direction of  gradient
In the case of lane lines, we're interested only in edges of a particular orientation. The function `dir_thresh` computes the direction, or orientation, of the gradient and applies a defined threshold.
The direction of the gradient is simply the inverse tangent (arctangent) of the `y` gradient divided by the `x` gradient:

![alt text][image2_4]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `original2bird_eye`.  This function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points were hardcoded and I used the following parameters:

```python
# Source matrix to define a Region Of Interest
src = np.array([[580, 460],
                [203, 720],
                [1127, 720],
                [705, 460]], dtype=np.float32)

# Target matrix to transform to
dst = np.array([[320, 0],
                [320, 720],
                [960, 720],
                [960, 0]], dtype=np.float32)
```

After defining the Region Of Interest, the next step is to warp the image using `original2bird_eye`, to see the image from bird's eye perspective. To do this we need to calculate a Matrix with the defined source and destination points: 

```python

trans_matrix = cv2.getPerspectiveTransform(src, dst) 
```
After calculating the Matrix we can now apply the OpenCV function `warpPerspective`  to get the final warped image:

![alt text][image3]

#### 4 Itentifying lane-line pixels and fitting their positions with a polynomial

##### 4.1 Creation of Histogram and finding its peaks
After applying calibration, thresholding, and a perspective transform to a road image, we now have a binary image where the lane lines stand out clearly so that the lane lines can be mapped out.
The first step of mapping out the lane lines is the create a Histogram of lower half of the image. 
```python

def get_histogram(warpedimage):
    return np.sum(warpedimage[warpedimage.shape[0]//2:,:], axis=0)
```
With this way we are able to find out a distinction between the left lane pixels and right lane pixels:
![alt text][image4]

##### 4.2 Sliding Window search

After finding the the two highest peaks from our histogram we can now use it as a starting point for determining where the lane lines are, and then use **sliding windows** moving upward in the image to determine where the lane lines go.

The sliding window is applied in following steps:

1. The left and right base points are calculated from the histogram
2. We then calculate the position of all non zero x and non zero y pixels.
3. We then Start iterating over the windows where we start from points calculate in point 1.
4. We then identify the non zero pixels in the window we just defined
5. We then collect all the indices in the list and decide the center of next window using these points
6. Once we are done, we seperate the points to left and right positions

##### 4.3 Fit a polynomial 
Now that we have found all our pixels belonging to each line through the sliding window method, it's time to fit a polynomial to the line using a second degree polynomial using with `np.polyfit`:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

If we summarize all the described points, we now have a thresholded image, estimation which pixels belong to the left and right lane lines (shown in red and blue, respectively, in the image above) and a fitted polynomial to those pixel positions.
The next step is computation of the radius of curvature of the fit.

The implemented function `calculate_curvature_radius` returns the calculated curvature and vehicle's position on the center
that we computed as follows: 
Radius:

1. First we define values to convert pixels to meters
2. Plot the left and right lines
3. Calculate the curvature from left and right lanes seperately
4. Return mean of values calculated in step 3.

Vehicle's position on the center:

1. Calculate the lane center by evaluating the left and right polynomials at the maximum Y and find the middle point.
2. Calculate the vehicle center transforming the center of the image from pixels to meters.
3. The sign between the distance between the lane center and the vehicle center gives if the vehicle is on to the left or the right.

#### 6.  Example image of result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_green_lane`. A polygon is generated based on plots of the left and right fits. The generated points are mapped back to the image space using the inverse transformation matrix generated by the perspective transformation in `bird_eye2original` function of my code. The image below is an example of the results of the `draw_green_lane` function:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Thoughts:
This project was a fascinating challenge that taught me a lot about CV techniques.
The implemented pipeline works fine on the project video and looks quite robust. This is becase there's litte changes in elevation, lighting or any steep bends. 
As seen in the challenge videos, changes in road conditions and colors present a problem and must have specific color filters. Also, other lines are picked up by the pipeline that match all the processing but are not the lane lines.
Similarly, during lane-change, lane-merging or any other situation where lanes are converging to normal pattern, the algorithm will fail.

