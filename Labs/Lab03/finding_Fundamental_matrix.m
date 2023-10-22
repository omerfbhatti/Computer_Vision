%%% STEREO GEOMETRY 
% Given P, P' --> find F, the Fundamental Matrix

% Camera Matrix
K = [960, 0, 960;
     0, 960, 540;
     0, 0, 1];
K_prime = K;

% Rotation Matrix between the two cameras
R = [1, 0, 0;
     0, 1, 0;
     0, 0, 1];

% Translation Matrix of the two cameras in the world coordinates
t = [0, 0, 0]';
t_prime = [-1, 0, -3]';

P = K * [R, t]
P_prime = K_prime * [R, t_prime]

% F = ep_prime_cross_matrix * Kp * R * inv(K)
% ep = epipolar point'
% C = Camera Center, Camera origin point in world coordinates

% ep_prime = P_prime * C

C = [-1, 0, -3, 1]';

ep_prime = P_prime * C;
ep_prime = ep_prime/ep_prime(3)

ep = P * [1,0,3,1]';
ep = ep/ep(3)

ep_prime_cross_matrix = [0, -ep_prime(3), ep_prime(2);
                         ep_prime(3), 0, -ep_prime(1);
                         -ep_prime(2), ep_prime(1), 0];
                     
F = ep_prime_cross_matrix * K_prime * R * inv(K);
F = F/F(2,1)

% Given point x, find epipolar line l_prime
x = [960, 540, 1]';
l_prime = F * x;
l_prime = l_prime/l_prime(3)

% Calculate epipolar line l for x_prime = x
% L = F' * x_prime
x_prime = [960, 540, 1]';
l = F' * x_prime;
l = l/l(3)
