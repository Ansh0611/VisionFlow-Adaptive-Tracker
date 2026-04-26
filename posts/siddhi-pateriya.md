### Uncovering 3D Trajectories

When the VisionFlow system detects that the camera itself is moving, object tracking becomes unreliable because the entire background is shifting. To handle this, we switch to Monocular Visual Odometry (VO), a fascinating technique that estimates the camera's trajectory through 3D space using only a sequence of 2D images. 

The pipeline begins by identifying strong visual anchors in the environment. We use the Shi-Tomasi corner detector to find distinct features like edges of buildings or textured surfaces. We then track these features across consecutive frames using the Lucas-Kanade optical flow algorithm, which calculates the apparent motion of these points as the camera moves.

The geometric heart of this process is the Essential Matrix. By finding corresponding points between two frames, we can compute this matrix, which mathematically encodes the relative rotation and translation of the camera between those two positions. Since feature matching is never perfect in the real world, we employ RANSAC (Random Sample Consensus) to robustly estimate the matrix while ignoring outliers and mismatches. 

Finally, we decompose the Essential Matrix to recover the camera's pose. By accumulating these frame-to-frame translations and rotations, we construct a continuous trajectory path. The beauty of Visual Odometry lies in its ability to reconstruct the unseen 3D world solely from flat pixels, allowing our system to maintain spatial awareness even when the camera is in motion.
