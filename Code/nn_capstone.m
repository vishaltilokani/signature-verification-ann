genuinePath = "D:\nn capstone\database\CEDAR\signatures\full_org";
forgedPath  = "D:\nn capstone\database\CEDAR\signatures\full_forg";

%%Step 1
% imageDatastore is a MATLAB tool that scans a folder and remembers
% the location of every image inside it, without loading them all
% into memory at once. This matters later when we work with hundreds
% of images at a time.
genuineDS = imageDatastore(genuinePath);
forgedDS  = imageDatastore(forgedPath);

% Count how many images were found in each folder
numGenuine = numel(genuineDS.Files);
numForged  = numel(forgedDS.Files);

fprintf("Genuine signatures found: %d\n", numGenuine);
fprintf("Forged signatures found:  %d\n", numForged);

% Quick sanity check — stop early with a clear message if something's wrong,
% instead of a confusing error later.
if numGenuine == 0 || numForged == 0
    error("No images found. Double-check that genuinePath and forgedPath point to the correct folders.");
end

% Show one genuine and one forged signature side-by-side.
% This proves MATLAB can actually READ the image files, not just count them.
genuineImg = readimage(genuineDS, 1);
forgedImg  = readimage(forgedDS, 1);

figure;
subplot(1,2,1);
imshow(genuineImg);
title("Sample Genuine Signature");

subplot(1,2,2);
imshow(forgedImg);
title("Sample Forged Signature");

fprintf("\nIf you can see two signature images above, your dataset is ready.\n");

%% STEP 2: Image Preprocessing

% Read first genuine signature
img = readimage(genuineDS, 1);

% Check image type
disp("Image class:");
disp(class(img));

disp("Image size:");
disp(size(img));

% Convert RGB image to grayscale only if necessary
if size(img,3) == 3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end

% Convert image to grayscale uint8 if necessary
if ~isa(grayImg, 'uint8')
    grayImg = im2uint8(grayImg);
end

% Display original and grayscale
figure;

subplot(1,2,1);
imshow(img);
title("Original Signature");

subplot(1,2,2);
imshow(grayImg);
title("Grayscale Signature");


%% STEP 3: Convert to Binary

% Convert grayscale image to binary
% Fallback binarization using image mean as threshold
if ~isfloat(grayImg)
    grayImg = im2double(grayImg);
end
th = mean(grayImg(:));
binaryImg = grayImg > th;

% Display binary image
figure;

subplot(1,2,1);
imshow(grayImg);
title("Grayscale");

subplot(1,2,2);
imshow(binaryImg);
title("Binary Signature");

%% STEP 4: Automatically Crop Signature

% Find the pixels containing the signature
[row, col] = find(~binaryImg);

% Check whether signature pixels were detected
if isempty(row) || isempty(col)
    error("No signature detected in the binary image.");
end

% Find bounding box
top = min(row);
bottom = max(row);
left = min(col);
right = max(col);

% Crop the signature
croppedImg = binaryImg(top:bottom, left:right);

% Display cropped signature
figure;
imshow(croppedImg);
title("Cropped Signature");


%% STEP 5: Resize Signature

% Resize cropped signature to 64 x 64 pixels
resizedImg = imresize(croppedImg, [64 64]);

% Display resized signature
figure;
imshow(resizedImg);
title("64 x 64 Signature");


