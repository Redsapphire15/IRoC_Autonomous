import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pyrealsense2 as rs

# Calibration Matrix for RealSense camera (replace with your actual calibration)
k = np.array([[518.56666108, 0., 329.45801792],
              [0., 518.80466479, 237.05589955],
              [0., 0., 1.]])

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                 minEigThreshold=1e-4)

# Shi-Tomasi corners detector are used
def corners(img):
    corners = cv.goodFeaturesToTrack(img, mask=None, maxCorners=400, qualityLevel=0.01, minDistance=7,
                                     blockSize=3, useHarrisDetector=False, k=0.04)
    return corners

# Tracking using Kanade-Lucas-Tomasi Optical flow tracker
def track_features(img1, img2, corners):
    p1, st, err = cv.calcOpticalFlowPyrLK(img1, img2, corners, None, **lk_params)
    p1 = p1[st == 1]  # Selecting good points
    corners = corners[st == 1]
    return p1, corners

# Triangulation for 3D point cloud estimation
def triangulaion(R, t, pt1, pt2, k):
    pr = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    pr_mat = np.dot(k, pr)
    P = np.hstack((R, t))
    P1 = np.dot(k, P)
    ch1 = np.array(pt1)
    ch2 = np.array(pt2)
    ch1 = pt1.transpose()
    ch2 = pt2.transpose()
    cloud = cv.triangulatePoints(pr_mat, P1, ch1, ch2)
    cloud = cloud[:4, :]
    return cloud

# Scale estimation
def scale(O_cloud, N_cloud):
    siz = min(O_cloud.shape[1], N_cloud.shape[1])
    o_c = O_cloud[:, :siz]
    n_c = N_cloud[:, :siz]
    o_c1 = np.roll(o_c, axis=1, shift=1)
    n_c1 = np.roll(n_c, axis=1, shift=1)
    scale = np.linalg.norm((o_c - o_c1), axis=0) / (np.linalg.norm(n_c - n_c1, axis=0) + 1e-8)
    scale = np.median(scale)
    return scale

# Main loop
try:
    prev_frame = None
    trans = np.array([[0], [0], [0]])
    rotation = np.eye(3)
    o_cloud = None
    n_cloud = None

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy arrays
        image = np.asanyarray(color_frame.get_data())

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        if prev_frame is not None:
            pts1 = corners(prev_frame)
            pts2, pts1 = track_features(prev_frame, gray, pts1)
            E, mask = cv.findEssentialMat(pts2, pts1, k, cv.RANSAC, prob=0.999, threshold=0.4, mask=None)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, k)
            n_cloud = triangulaion(R, t, pts1, pts2, k)
            sc = scale(o_cloud, n_cloud)
            trans = trans - sc * np.dot(rotation, t)
            rotation = np.dot(rotation, R)
            x1 = trans[0]
            z1 = trans[2]
            plt.scatter(x1, z1, c='green', s=3)
            plt.draw()
            plt.pause(0.01)

        prev_frame = gray
        o_cloud = n_cloud

except KeyboardInterrupt:
    # Stop streaming
    pipeline.stop()
    plt.show()
