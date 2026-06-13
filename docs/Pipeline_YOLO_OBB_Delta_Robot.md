# Pipeline for Deploying YOLO OBB in a Conveyor Belt PCB Sorting System

This document standardizes the 5-stage pipeline for implementing a Computer Vision solution to classify and detect 3 types of PCBs using YOLO11-OBB. It covers the inference system on the PC and the real-time communication with the PLC to control a Delta Robot. The parameters are optimized for a conveyor belt running at a speed of 20 cm/s.

---

## Stage 1: Hardware Setup & Raw Data Preparation

**1. Hardware Setup & Camera Calibration**
* **Intrinsic Calibration:** Use a checkerboard to obtain the camera matrix and apply undistortion, ensuring that PCBs at the edges of the frame are not distorted compared to those in the center.
* **Extrinsic Calibration:** Determine the Pixel-to-mm ratio (e.g., 1 pixel = 0.5 mm) and synchronize the camera's center coordinate system with the robotic arm's base coordinate system.
* **Hardware Configuration:** Set a fast Shutter Speed (e.g., 1/500s) combined with high-intensity industrial LED lighting to eliminate motion blur at the 20 cm/s conveyor speed.

**2. Data Collection**
* Run the conveyor belt at the actual operational speed (20 cm/s) and use a Python script to extract frames from the video stream.
* Collect approximately 300 - 500 raw images for each PCB class, capturing natural variations: minor mechanical vibrations, ambient lighting changes, and random orientations.

---

## Stage 2: Pre-processing & Annotation

**3. OBB Annotation & Format Conversion**
* Use an annotation tool (such as Label Studio) to draw Oriented Bounding Boxes (OBB).
* Export the labels to the YOLO OBB standard format: `[class, x_center, y_center, width, height, theta]` or the 4-corner format `[x1, y1, x2, y2, x3, y3, x4, y4]`.

**4. Data Augmentation & Noise Addition**
* Configure parameters to compensate for mechanical tolerances:
    * Rotation: 360 degrees (since PCBs will land in random orientations).
    * Translation: ~10%.
    * Scale: ±3% to ±5% to compensate for conveyor vibration.
    * Add Noise: Artificial motion blur and light glare reflecting off solder pads.
* *Note:* Strictly disable heavy geometric distortions like Perspective or Shear.

**5. Dataset Split**
* Split the dataset using the standard ratio: Train (70%), Validation (20%), and Test (10%).

---

## Stage 3: Training

**6. Training Configuration**
* Set up the `data.yaml` file to define dataset paths and the 3 PCB classes.
* Fine-tune the `hyp.yaml` file: Adjust learning rate, batch size, and online augmentation parameters.

**7. Model Training**
* Utilize pre-trained weights like `yolo11n-obb.pt` (Nano version) or `yolo11s-obb.pt` for Transfer Learning. This ensures optimal inference speed and hardware efficiency.

---

## Stage 4: Evaluation & Optimization (Inference Server)

**8. Evaluation & Error Analysis**
* Monitor metrics such as `mAP50` and `mAP50-95`.
* Pay special attention to the angle error ($\theta$), as even a slight angular deviation can cause the end-effector (gripper/suction cup) to damage the electronic components.
* Test the model on real-world video streams to verify FPS and bounding box stability (jitter).

**9. Model Export & Optimization**
* **Mandatory:** Export the trained model to a hardware-optimized format to achieve Real-time processing speeds.
    * NVIDIA GPU: Export to **TensorRT** (`.engine`).
    * Intel CPU: Export to **OpenVINO**.
* This step drastically reduces inference latency from ~30ms down to 3-5ms.

---

## Stage 5: System Integration & PLC Communication

**10. Data Flow Processing (Python Inference Pipeline)**
* **Step 1 - Capture & Undistort:** Read frames from the camera and apply `cv2.undistort()`.
* **Step 2 - Inference:** Run the TensorRT model to extract the pixel center $x_{img}, y_{img}$, dimensions $w, h$, and angle $\theta$.
* **Step 3 - Coordinate Transformation:** Apply the extrinsic transformation matrix to convert from Pixels to the Robot's Millimeter coordinate system:
    $$\begin{bmatrix} X_{robot} \\ Y_{robot} \\ 1 \end{bmatrix} = M_{calib} \times \begin{bmatrix} x_{img} \\ y_{img} \\ 1 \end{bmatrix}$$
* **Step 4 - Object Tracking:** Establish a virtual "Trigger Line" across the camera frame. The coordinates are locked and transmitted only when the center of the PCB crosses this line, preventing duplicate signals for the same object.

**11. PC - PLC Communication & Real-time Conveyor Tracking**
* **Communication:** Use Native TCP/IP Sockets or Modbus TCP/S7comm to transmit the coordinate string `[X_mm, Y_mm, Angle_degree]` from the PC to the PLC.
* **Conveyor Tracking (Synchronized on the PLC):**
    * The PLC reads pulses from the Encoder attached to the conveyor shaft via a High-Speed Counter (HSC).
    * Store the received coordinates from the PC into a FIFO Queue, paired with the Encoder value at the exact moment the image was captured.
    * Continuously calculate the real-time translating position:
        $$Current\_Position = X_{robot} + (Current\_Encoder - Captured\_Encoder) \times Pulse\_Resolution$$
    * Once the dynamically tracked coordinate enters the Robot Delta's Workspace, the PLC triggers the Servo motors to execute the pick-and-place cycle using the provided angle $\theta$.