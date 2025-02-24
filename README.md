# Human Pose Estimation Using OpenCV and Deep Learning

This project demonstrates human pose estimation using a deep learning model with OpenCV. The code takes an image or video as input and detects human body poses by identifying key points on the human body such as the nose, shoulders, elbows, wrists, hips, knees, and ankles.

## Prerequisites

Before running the code, make sure you have the following installed:

- **Python 3.x**
- **OpenCV 4.x** with DNN module support
- **Numpy**
- **Matplotlib**

You can install the necessary dependencies using pip:

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

## Getting Started

Follow the steps below to use this project:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/human-pose-estimation.git
cd human-pose-estimation
```

### 2. Download the Pretrained Model

The code uses a pretrained TensorFlow model (`graph_opt.pb`) for pose estimation. You can download the model from the official repository or any source providing this model. Once downloaded, place it in the root directory of this project.

- **Pretrained Model File**: [Download `graph_opt.pb`](https://example.com/graph_opt.pb)

### 3. Prepare Input Data

#### Image-Based Pose Estimation
Prepare an input image (`stand.jpg`) of a person that you want to perform pose estimation on. Place the image in the same directory as the code or specify the full path in the code.

#### Video-Based Pose Estimation
For video-based pose estimation, use the provided `run.mov` file or any other video file you want to process.

### 4. Run the Code

#### Running on an Image
```bash
python pose_estimation.py
```

#### Running on a Video
```bash
python pose_estimation_Video.py
```

### 5. Output

After running the script, an output image with drawn body pose landmarks and connections will be saved as `OutPut-image.png`. For video processing, an output video file will be generated.

## Code Explanation

### Key Concepts

- **BODY_PARTS**: A dictionary that maps human body parts (such as "Nose", "Neck", "RShoulder") to specific indices in the output of the neural network.
- **POSE_PAIRS**: A list of pairs of body parts that should be connected by lines (e.g., ["Neck", "RShoulder"] connects the neck to the right shoulder).
- **Pose Detection**: The function `poseDetector(frame)` performs the following steps:
  - Reads the input image.
  - Preprocesses the image for the deep learning model.
  - Feeds the image to the pre-trained model.
  - Extracts body parts and their coordinates from the model's output.
  - Draws key points and skeleton lines on the input image.

### Model Loading

The model is loaded using the OpenCV DNN module:

```python
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
```

This loads the TensorFlow model for human pose estimation.

### Pose Detection

The function `poseDetector(frame)` handles the detection and visualization of human poses. It works by passing the input image through the model, extracting body part coordinates, and drawing lines and points to represent the detected pose.

### Visualization

The body parts and their connections are drawn on the input image using OpenCV’s `cv2.line()` and `cv2.ellipse()` functions.

### Example

#### Running Pose Estimation on an Image
```python
input = cv2.imread('stand.jpg')
output = poseDetector(input)
cv2.imwrite("OutPut-image.png", output)
```

#### Running Pose Estimation on a Video
```python
cap = cv2.VideoCapture('run.mov')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = poseDetector(frame)
    cv2.imshow("Pose Estimation", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### Output

After running the script, the output image will be saved with key points and skeleton lines:

- **Red circles** represent the detected key body parts (e.g., nose, shoulders).
- **Green lines** represent the connections between these body parts.

For videos, the output video file will have the detected poses drawn on each frame.

## Conclusion

This project shows how human pose estimation can be done using a pre-trained TensorFlow model and OpenCV's DNN module. It can be extended to real-time pose estimation using a webcam or video stream by modifying the input and output handling.

