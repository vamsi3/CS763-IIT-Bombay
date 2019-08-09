%{
Joint Indices
1  - Right ankle
2  - Right knee
3  - Right hip
4  - Left hip
5  - Left knee
6  - Left ankle
7  - Hip
8  - Spine
9  - Neck
10 - Head
11 - Right wrist
12 - Right elbow
13 - Right shoulder
14 - Left shoulder
15 - Left elbow
16 - Left wrist

Kinematic Chain Configuration (with root as hip)
Joint Index, Parent Joint Index
3,7    - Right hip, Hip
2,3    - Right knee, Right hip
1,2    - Right ankle, Right knee
4,7    - Left hip, Hip
5,4    - Left knee, Left hip
6,5    - Left ankle, left hip
8,7    - Spine, Hip
9,8    - Neck, Spine
10,9   - Head, Neck
13,9   - Right shoulder, Neck
12,13  - Right elbow, Right shoulder
11,12  - Right wrist, Right elbow
14,9   - Left shoulder, Neck
15,14  - Left elbow, Left shoulder
16,15  - Left wrist, Left elbow
%}

n_joints = 16;

kinematic_chain = [
    3,7; % right hip
    2,3; % right upper leg
    1,2; % right lower leg 
    4,7; % left hip
    5,4; % left upper leg
    6,5; % left lower leg
    8,7; % lower spine
    9,8; % upper spine
    10,9; % neck-head
    13,9; % right shoulder bone
    12,13; % right upper arm
    11,12; % right lower arm
    14,9; % left shoulder bone
    15,14; % left upper arm
    16,15 % left lower arm
];

n_parts = size(kinematic_chain,1);

hip_idx = 7;
% A base pose in XZ plane (chosen for consistency between display and joints)
base_pose = [
  [1,0,-5]; % right ankle  
  [1,0,-3]; % right knee
  [1,0,0]; % right hip
  [-1,0,0]; % left hip
  [-1,0,-3]; % left knee
  [-1,0,-5]; %left ankle
  [0,0,0]; % hip
  [0,0,2]; % midspine
  [0,0,4]; % neck
  [0,0,6]; % head
  [5,0,4]; % right wrist
  [3,0,4]; % right elbow
  [1,0,4]; % right shoulder
  [-1,0,4]; % left shoulder
  [-3,0,4]; % left elbos
  [-5,0,4]; % left wrist
];

% Plot pose and check
plotPose(base_pose, kinematic_chain);
w = waitforbuttonpress;
rot_angles = zeros(n_parts, 3);
rot_mat = angles2rot(rot_angles);

% Test Case 0: Identity rotation matrix
result_pose = transformPose(rot_mat, base_pose(:,:), kinematic_chain, hip_idx);
plotPose(result_pose, kinematic_chain);
w = waitforbuttonpress;

% Test Case 1: Sitting
% rot_angles = zeros(n_parts, 3);
% rot_angles(2,:) = [-90,0,0];
% rot_angles(3,:) = [90,0,0];
% rot_angles(5,:) = [-90,0,0];
% rot_angles(6,:) = [90,0,0];
% rot_angles(11,:) = [0,0,-90];
% rot_angles(12,:) = [0,-90,0];
% rot_angles(14,:) = [0,0,90];
% rot_angles(15,:) = [0,90,0];
% rot_angles1 = rot_angles;
load('rot_angles.mat');
rot_mat = angles2rot(rot_angles);
result_pose = transformPose(rot_mat, base_pose(:,:), kinematic_chain, hip_idx);
plotPose(result_pose, kinematic_chain);
w = waitforbuttonpress;


% Test Case 2: Forward split
load('rot_matrix.mat');
result_pose = transformPose(rot_mat, base_pose(:,:), kinematic_chain, hip_idx);
plotPose(result_pose, kinematic_chain);

