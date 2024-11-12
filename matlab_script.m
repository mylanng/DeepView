% Load the point cloud from .ply file
pointCloudData = pcread('scan_001.ply');

% Visualize the point cloud
figure;
pcshow(pointCloudData);
title('Loaded 3D Mesh Point Cloud');
xlabel('X'); ylabel('Y'); zlabel('Z');

% If you need to denoise or smooth the point cloud
smoothedCloud = pcdenoise(pointCloudData);
figure;
pcshow(smoothedCloud);
title('Denoised Point Cloud');

%% 
% Define the file path to your volumetric data
volumeFolder = 'path_to_volume_folder';

% Create a custom datastore to read .raw volumes
trainVolData = fileDatastore(volumeFolder, 'ReadFcn', @(filename) loadRawVolume(filename, [768, 768, 1280]), 'FileExtensions', '.raw');

% Define function to load and reshape .raw volume
function data = loadRawVolume(filename, volumeSize)
    fid = fopen(filename, 'rb');
    data = fread(fid, prod(volumeSize), 'float32'); % Adjust data type if different
    fclose(fid);
    data = reshape(data, volumeSize); % Reshape to 3D volume
end

% If labels are also in .raw format, set up a similar datastore for trainLabels
