# Dataset Preparation and Augmentation Strategy

## 1. Dataset Collection and Base Quantities
* **Base Images (Raw Data):** Images must be captured directly on the actual green conveyor belt to account for background camouflage and real-world industrial lighting.
* **Quantity per Class:** 300 to 500 raw images for each of the 3 PCB types.
* **Total Base Dataset:** Approximately 900 to 1,500 raw images.

## 2. Types of Dataset Variations (Real and Synthetic)
To ensure the YOLO-OBB model performs robustly in the actual industrial environment, the dataset must incorporate the following specific categories of image variations:
## 2.1. Percentage Distribution of Variations (Augmentation Mix)
Within the augmented Training set (which makes up 70% of the total dataset), the distribution of variations should be allocated as follows to optimize for the 20 cm/s conveyor speed and robotic picking precision:

* **Background Camouflage (Green-on-Green): 100%**
  * *Reasoning:* Every single base image must be captured on the actual green conveyor belt. The model must consistently learn to ignore the background across the entire dataset.
* **Orientation Variations (360° Yaw Rotation): 80% - 100%**
  * *Reasoning:* Because the Delta robot needs precise `theta` coordinates for any possible angle, almost every augmented image should be randomly rotated. This prevents the model from being biased toward upright or horizontal orientations.
* **Translation & Scaling (Vibration Compensation): ~50%**
  * *Reasoning:* About half of the training images should include a ±10% position shift or ±5% scale. This ensures the model detects PCBs accurately even if they are at the very edge of the camera frame or if the belt vibrates vertically.
* **Motion Blur: ~30%**
  * *Reasoning:* While the camera's shutter speed (1/500s) minimizes blur, mechanical jerks at 20 cm/s will still cause occasional edge softening. Applying artificial motion blur to 30% of the dataset ensures the model does not drop detections during high-speed movement.
* **Lighting Variations (Glare, Shadows, Brightness): ~20% - 25%**
  * *Reasoning:* Applied to simulate fluctuations in factory lighting or the shadow of the robot arm moving overhead. This prevents the model from over-fitting to a specific light intensity.
* **Noise & Partial Occlusion: ~5% - 10%**
  * *Reasoning:* A small fraction of the dataset should include artificial noise (Gaussian noise) or minor cutouts (simulating dust on the lens or small debris on the belt) to improve the overall robustness of the model.
* **Lighting Variations (Ánh sáng):**
    * **Glare & Reflections:** High-intensity light reflecting off metallic solder pads, vias, and IC pins caused by industrial LED lighting.
    * **Shadows:** Dynamic shadows cast across the PCBs by the moving Delta robot arm or shifts in ambient factory lighting.
* **Orientation & Position Variations (Xoay & Vị trí):**
    * **360° Yaw Rotation:** PCBs placed at completely random angles ($0^\circ$ to $360^\circ$) on the belt. This is critical to train the OBB algorithm to predict the `theta` angle precisely for the robotic end-effector.
    * **Edge Translation:** PCBs positioned near the extreme edges of the camera's Field of View (FoV).
* **Motion Blur Variations (Mờ do chuyển động):**
    * Images containing natural or synthetically generated motion blur. Since the conveyor operates at a continuous speed of **20 cm/s**, the camera frames will naturally exhibit slight edge softening. The model must learn to detect PCBs despite this blur.
* **Background Camouflage (Nền trùng màu):**
    * "Green-on-Green" scenarios where the PCB soldermask blends with the green rubber conveyor belt. The dataset must force the model to prioritize structural features (white silkscreen, geometric edges, metallic contrast) over color recognition.
* **Partial Occlusion & Noise (Che khuất & Nhiễu):**
    * Minor visual obstructions such as dust on the lens, stray wires on the belt, or sensor noise in low-light conditions.

## 3. Data Augmentation Strategy (YOLO Hyperparameters)
To achieve the variations mentioned above without manually photographing thousands of setups, the base dataset is synthetically expanded to a total of **3,000 - 5,000 images** using the following specific augmentation settings:
* **Rotation:** `degrees: 180.0` (Allows rotation from -180° to 180°).
* **Translation:** `translate: 0.1` (Shifts the image by ±10% horizontally/vertically).
* **Scaling:** `scale: 0.05` (Scales the image by ±5% to compensate for vertical Z-axis vibrations of the camera mount or belt).
* **Blur:** `blur: 0.1` (Adds artificial Gaussian/Motion blur to mimic the 20 cm/s movement).
* **Color Space Jitter:** `hsv_h`, `hsv_s`, `hsv_v` adjustments to simulate different lighting intensities and shadow effects.
* *(Note: Geometric distortions like `perspective` and `shear` are strictly set to 0.0, as PCBs are rigid 2D planes).*

## 4. Dataset Splitting Ratio (70/20/10)
The fully prepared dataset is divided into three distinct subsets to ensure reliable model evaluation:
* **Training Set (70%):** Used to optimize the model weights. This set contains the highest amount of augmented variations to force the model to learn geometry and texture rather than background colors.
* **Validation Set (20%):** Evaluated simultaneously during the training process. It is used to monitor the loss function, tune hyperparameters, and prevent overfitting.
* **Testing Set (10%):** Completely unseen data. To accurately benchmark real-world reliability (especially the precision of the coordinate mapping for the PLC), this set must consist **strictly of real, unaugmented images** captured directly from the final camera setup while the belt is running.