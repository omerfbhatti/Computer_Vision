import numpy as np
import cv2


frame1 = cv2.imread('Final-Image-1.png')
frame2 = cv2.imread('Final-Image-2.png')
K = np.array([[7.96607443e+02, 0.00000000e+00, 1.00207058e+03],
                [0.00000000e+00, 7.93970516e+02, 4.25696490e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeff = np.array([-1.53874001e-01, 2.36318704e-02, 6.20711711e-05,
                        1.96694600e-04, -1.66714132e-03])

frame1_u = cv2.undistort(frame1, K, dist_coeff)
frame2_u = cv2.undistort(frame2, K, dist_coeff)
cv2.imwrite('frame1u.png', frame1_u)
cv2.imwrite('frame2u.png', frame2_u)


# Draw Feature Points
akaze = cv2.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(frame1_u, None)
kpts2, desc2 = akaze.detectAndCompute(frame2_u, None)
keypts_image_1 = frame1_u.copy()
keypts_image_2 = frame2_u.copy()
keypts_image_1 = cv2.drawKeypoints(frame1_u, kpts1, keypts_image_1, None,
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypts_image_2 = cv2.drawKeypoints(frame2_u, kpts1, keypts_image_2, None,
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('frame1uk.png', keypts_image_1)
cv2.imwrite('frame2uk.png', keypts_image_2)


# Find E Matrix and inlier points
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)
matched1 = []
matched2 = []
matched_pts_1 = []
matched_pts_2 = []
matches = []
nn_match_ratio = 0.8
for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        matched_pts_1.append(kpts1[m.queryIdx].pt)
        matched_pts_2.append(kpts2[m.trainIdx].pt)
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])
        matches.append(m)
        
E, inlier_mask = cv2.findEssentialMat(np.array(matched_pts_1), np.array(matched_pts_2), K, method=cv2.RANSAC)
inlier_matches = []
for i in range(len(inlier_mask)):
    if inlier_mask[i]:
        inlier_matches.append(matches[i])

out_img = frame1_u.copy()
out_img = cv2.drawMatches(frame1_u, kpts1, frame2_u, kpts2, inlier_matches, out_img)
cv2.imwrite('inlier-matches.png', out_img)


# Recover Pose
retval, R, t, mask = cv2.recoverPose(E, np.array(inlier_pts1),
np.array(inlier_pts2), K)
twist, _ = cv2.Rodrigues(R)


# Triangulate 3d points
P1 = K @ np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = K @ np.concatenate((R, t), 1)
points = cv2.triangulatePoints(P1, P2, np.array(inlier_pts1).T,
np.array(inlier_pts2).T)
points = points[0:3,:] / points[3:4,:]



