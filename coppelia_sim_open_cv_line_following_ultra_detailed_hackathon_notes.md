# CoppeliaSim + OpenCV Line Following  
## Ultra-Detailed Hackathon Notes (Simulation + Theory + Debugging)

> This document is a single source of truth for understanding and implementing  
> vision-based line following using OpenCV in CoppeliaSim.  
> Designed for screening tests, hackathons, and competitions.

---

## 1. Sanity Checks in CoppeliaSim (CRITICAL)

### Problem
Robot keeps moving even after simulation is stopped and restarted.

### Root Cause
CoppeliaSim remembers previous joint target velocities.

If target velocity ≠ 0 after stopping the simulation, the robot will continue moving.

### Correct Procedure
1. Stop the simulation
2. Double-click the robot
3. Open Joint → Dynamic Properties
4. Ensure:
   - Target Velocity = 0
5. Restart simulation

**Golden Rule**  
If the robot moves unexpectedly, always check joint target velocities first.

---

## 2. Differential Drive – Practical Understanding

### Robot Model
- Differential drive (bus-like robot)
- Both wheels:
  - Positive velocity → forward
  - Negative velocity → backward

### Turning Logic

| Motion | Left Wheel | Right Wheel |
|------|-----------|------------|
| Straight | Same speed | Same speed |
| Turn Left | Slower | Faster |
| Turn Right | Faster | Slower |

**Key Insight**  
Turning is achieved by speed difference, not reversing wheels.

---

## 3. Core Robotics Loop

Every robot follows this loop:


### Applied to This Project

| Stage | Description |
|------|------------|
| Sense | Camera provides pixel data |
| Perceive | HSV thresholding, masking |
| Decide | Centroid error calculation |
| Act | Adjust motor speeds |

---

## 4. Sensing vs Perception

### Sensing
- Raw data acquisition
- Camera outputs numbers
- Robot does NOT see

### Perception
- Extracting meaning from data
- Detecting line position, centroid

**Interview Line**  
Robots don’t see objects, they interpret data.

---

## 5. Vision Sensor Data in CoppeliaSim

### Camera Output
- Resolution: 320 × 240
- Data format:
  - 1D byte array
  - RGBRGBRGB...

### Required Processing
1. Convert bytes → numeric array
2. Reshape to (height, width, 3)
3. Flip vertically (image is inverted)
4. Convert RGB → BGR (OpenCV standard)

If flipping is skipped, the image appears upside down.

---

## 6. Why RGB Fails and HSV Works

### RGB Problem
Same red object under different lighting produces different RGB values.

### HSV Advantage
Separates color identity from lighting.

| Component | Meaning |
|----------|--------|
| Hue | Color |
| Saturation | Intensity |
| Value | Brightness |

**Hackathon Rule**  
Never use RGB for real-world color detection.

---

## 7. HSV Range for Red Color

HSV Hue range: 0 → 180

Red appears in two regions:
- 0 → 15
- 165 → 180

### Mask Creation
Mask1: [0, 15]  
Mask2: [165, 180]  
Final Mask = Mask1 OR Mask2

---

## 8. Thresholding in OpenCV

### What Thresholding Does
- Pixel in range → WHITE
- Pixel out of range → BLACK

### Result
- Red line → white
- Everything else → black

This simplifies centroid calculation.

---

## 9. Region of Interest (ROI) Optimization

### Observation
The line exists mainly in the bottom half of the image.

### Optimization
Ignore upper half:
mask[0:h//2, :] = 0


### Benefits
- Faster computation
- Reduced noise
- Stable centroid

---

## 10. Centroid Calculation (Key Math)

### Objective
Find the horizontal position of the line.

### Formula (Center of Mass)
x_centroid = Σ(p_i · x_i) / Σ(p_i)

### OpenCV Implementation

M = cv2.moments(mask)
cx = M['m10'] / M['m00']


| Term | Meaning |
|-----|--------|
| m00 | Total pixel weight |
| m10 | Weighted x-sum |

---

## 11. Error Calculation

### Definitions
image_center = width // 2
error = cx - image_center


| Error Sign | Interpretation |
|------------|---------------|
| Positive | Line on right |
| Negative | Line on left |

---

## 12. P-Controller Steering

### Control Law
steering = -Kp × error


### Wheel Velocities
left_wheel = base_speed - steering
right_wheel = base_speed + steering


### Tuning Guidelines
- High Kp → oscillations
- Low Kp → sluggish response
- Always tune at low speed first

---

## 13. Handling Line Loss

### Simple Approach
- Stop
- Reset centroid
- Wait

### Advanced Approach (Competition Grade)
- Predict arc
- Continue last known direction
- Search until line reappears

---

## 14. Intersections Logic

### Detection
- Both sensors fail
- Centroid disappears

### Common Strategies
1. Always turn left
2. Always turn right
3. Memory-based path
4. Rotate ±90° and scan

**Competition Tip**  
Simple logic beats complex logic under time pressure.

---

## 15. IR Sensor vs Vision Sensor

### IR Sensor
- Reflectance-based
- Black → no reflection
- White → reflection

Logic:
If no reflection → turn


### Vision Sensor
- Handles curves and intersections
- Computationally heavier
- More flexible

---

## 16. Try–Except for Simulation Safety

### Problem
Ctrl+C stops code but simulation keeps running.

### Solution
try:
while True:
...
except KeyboardInterrupt:
stop_motors()
sim.stopSimulation()


Always reset motor velocities to zero.

---

## 17. Debugging Strategy (Hackathon Gold)

### Tools
- Print error values
- Display mask using imshow
- Visualize centroid
- Log wheel speeds

### Debug Mindset
Predict failure before it happens.

---

## 18. Real-World Hardware Issue: Battery Drop

### Problem
Battery voltage drops → motor speed changes.

### Solution
Buck Converter:
- Provides constant voltage
- Adjustable output
- Essential for consistent performance

---

## 19. Important Sensors for Screening Tests

| Sensor | Purpose |
|-------|---------|
| Encoder | Wheel rotation |
| IMU | Orientation, angular velocity |
| Camera | Vision |
| IR Sensor | Line detection |
| Ultrasonic | Distance |
| LiDAR | 3D mapping |
| Battery Sensor | Voltage monitoring |

---

## 20. Screening Test Breakdown

### Topics Covered
- Basic physics
- Logical reasoning
- Sensors
- PWM
- Pseudocode logic
- CoppeliaSim basics

### Format
- MCQ
- 30 minutes
- Easy to moderate difficulty

Focus is on logical thinking, not syntax.

---

## Final Advice for Hackathons

- Keep speed slow initially
- Debug visually
- Always reset velocities
- Simple logic > fancy logic
- Stability wins over speed

---

## Summary

You learned:
- Vision-based line following
- HSV thresholding
- Centroid math
- P-controller steering
- Simulation debugging
- Competition-ready robotics logic
