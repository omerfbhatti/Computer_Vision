def draw_epipolar(frame, l, K):
    p1 = K @ [[-100.0], [-l[0,0] * -100 / l[1,0] - l[2,0] / l[1,0]], [1.0]]
    p2 = K @ [[100.0], [-l[0,0] * 100 / l[1,0] - l[2,0] / l[1,0]], [1.0]]
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0,255))

Kinv = np.linalg.inv(K)
point1 = Kinv @ np.array([[1820], [100], [1]])
point2 = Kinv @ np.array([[1820], [980], [1]])
l1 = E @ point1
l2 = E @ point2
point1p = np.array([[1.0], [-l1[0,0] * 1.0 / l1[1,0] - l1[2,0] / l1[1,0]],
[1.0]])
point2p = np.array([[1.0], [-l2[0,0] * 1.0 / l2[1,0] - l2[2,0] / l2[1,0]],
[1.0]])
l1p = E.T @ point1p
l2p = E.T @ point2p
frame1_e = frame1_u.copy()
frame2_e = frame2_u.copy()
draw_epipolar(frame1_e, l1p, K)
draw_epipolar(frame1_e, l2p, K)
draw_epipolar(frame2_e, l1, K)
draw_epipolar(frame2_e, l2, K)
cv2.imwrite('frame1_e.png', frame1_e)
cv2.imwrite('frame2_e.png', frame2_e)
 
