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

X = rand(8,3) + [0,0,7];
X = [X, ones(8,1)]';

x_i = (P*X)';
x_prime = (P_prime*X)';

x_i = x_i./x_i(:,3)
x_prime = x_prime./x_prime(:,3)

A=[];
for i=1:8
  x = x_i(i,1);
  y = x_i(i,2);
  xp = x_prime(i,1);
  yp = x_prime(i,2);
  A = [A; [xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, 1]];
endfor

A;

% Find Fundamental Matrix
F = reshape(null(A), 3,3);
F = F/F(2,1)

% Find Essential Matrix
E = K_prime' * F * K
[U, W, V] = svd(E);
Vt=V';

W = [0, -1, 0;
     1, 0, 0;
     0, 0, 1];
     
u3 = U(:,3);

% Reconstructing P_prime from the Essential Matrix
recon_P_prime_1 = [U*W*Vt, u3];
recon_P_prime_2 = [U*W*Vt, -u3];
recon_P_prime_3 = [U*W'*Vt, u3];
recon_P_prime_4 = [U*W'*Vt, -u3];

if (recon_P_prime_1(1,1)<0)
  recon_P_prime_1 = -recon_P_prime_1;
endif

if (recon_P_prime_2(1,1)<0)
  recon_P_prime_2 = -recon_P_prime_2;
endif

if (recon_P_prime_3(1,1)<0)
  recon_P_prime_3 = -recon_P_prime_3;
endif

if (recon_P_prime_4(1,1)<0)
  recon_P_prime_4 = -recon_P_prime_4;
endif

recon_P = [eye(3), [0,0,0]']
recon_P_prime_1
recon_P_prime_2
recon_P_prime_3
recon_P_prime_4
P_prime_norm = inv(K) * P_prime
P_prime_norm_unit_translation = P_prime_norm;
P_prime_norm_unit_translation(:,4) = P_prime_norm(:,4)/norm( P_prime_norm(:,4) )