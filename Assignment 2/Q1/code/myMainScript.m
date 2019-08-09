% Q1data contains
% image_points 		: 2x6 double - Contains six 2D points in terms of pixel coordinates as captured by the camera.
% object_points 	: 3x6 double - Contains six 3D points in terms of checkboard coordinates (origin -> top corner of papers along wall edge; left & right wall -> X and Y axis directions respectively)
load('../input/Q1data.mat');

% Bullet 1
[normalized_image_points, T] = normalizePoints(image_points);
[normalized_object_points, U] = normalizePoints(object_points);

% Bullet 2
P_hat = DLT(normalized_image_points, normalized_object_points);
P = inv(T) * P_hat * U;

% Bullet 3
H = P(1:3, 1:3);
H_inv = inv(H);
X_o = - H_inv * P(:, 4);
[R, K] = qr(H_inv);
R = R';
K = inv(K);
P = P / K(3, 3);
K = K / K(3, 3);

% We can verify the above decomposition of P using: [this should give 1]
% sum(sum(abs(P - K * R * [eye(3, 3) -X_o]) > 0.00000000001)) == 0

% Bullet 4
num_points = size(image_points, 2);
predicted_points = P * [object_points; ones(1, num_points)];
predicted_points = predicted_points ./ (predicted_points(3,:));
predicted_points = predicted_points(1:2, :);
RMSE = norm(predicted_points - image_points) / 6;

% Visualization (not done on the actual image because the points were not properly visible. I felt this visualization gives a more clear picture on calibration accuracy)
hold on;
for i=1:num_points
    plot(image_points(1, i), image_points(2, i), 'r*');
    plot(predicted_points(1, i), predicted_points(2, i), 'b+');
end
legend('Image Points','Predicted Points');
hold off;
