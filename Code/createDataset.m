%% STEP 7: Create Improved Image-Based Feature Dataset
% Intelligent Signature Verification and Forgery Detection System

clc;
clear;

%% Paths

genuinePath = "D:\nn capstone\database\CEDAR\signatures\full_org";
forgedPath  = "D:\nn capstone\database\CEDAR\signatures\full_forg";

%% Load image datasets

genuineDS = imageDatastore(genuinePath);
forgedDS  = imageDatastore(forgedPath);

numGenuine = numel(genuineDS.Files);
numForged  = numel(forgedDS.Files);

fprintf("Genuine images: %d\n", numGenuine);
fprintf("Forged images: %d\n", numForged);

%% Image size

imageSize = [32 32];

numFeatures = imageSize(1) * imageSize(2);

%% Create feature matrix

X = zeros(numGenuine + numForged, numFeatures);

%% Create labels

Y = zeros(numGenuine + numForged, 1);

%% Process Genuine Signatures

fprintf("\nProcessing genuine signatures...\n");

for i = 1:numGenuine

    %% Read image

    img = readimage(genuineDS, i);

    %% Convert to grayscale

    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    %% Convert to binary

    grayDouble = double(grayImg);

    threshold = mean(grayDouble(:));

    % Signature = 1
    binaryImg = grayDouble < threshold;

    %% Find signature pixels

    [row, col] = find(binaryImg);

    %% Crop signature

    if ~isempty(row) && ~isempty(col)

        top = min(row);
        bottom = max(row);
        left = min(col);
        right = max(col);

        croppedImg = binaryImg(top:bottom, left:right);

    else

        croppedImg = binaryImg;

    end

    %% Preserve aspect ratio while resizing

    [h, w] = size(croppedImg);

    scale = min(28 / h, 28 / w);

    newH = max(1, round(h * scale));
    newW = max(1, round(w * scale));

    resizedSmall = imresize(croppedImg, [newH newW]);

    % Create 32x32 blank canvas
    finalImg = zeros(32, 32);

    % Center the signature
    startRow = floor((32 - newH) / 2) + 1;
    startCol = floor((32 - newW) / 2) + 1;

    finalImg(startRow:startRow+newH-1, ...
             startCol:startCol+newW-1) = resizedSmall;

    %% Convert to feature vector

    featureVector = double(finalImg(:))';

    X(i,:) = featureVector;

    %% Genuine = 1

    Y(i) = 1;

end

%% Process Forged Signatures

fprintf("Processing forged signatures...\n");

for i = 1:numForged

    %% Read image

    img = readimage(forgedDS, i);

    %% Convert to grayscale

    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    %% Convert to binary

    grayDouble = double(grayImg);

    threshold = mean(grayDouble(:));

    % Signature = 1
    binaryImg = grayDouble < threshold;

    %% Find signature pixels

    [row, col] = find(binaryImg);

    %% Crop signature

    if ~isempty(row) && ~isempty(col)

        top = min(row);
        bottom = max(row);
        left = min(col);
        right = max(col);

        croppedImg = binaryImg(top:bottom, left:right);

    else

        croppedImg = binaryImg;

    end

    %% Preserve aspect ratio while resizing

    [h, w] = size(croppedImg);

    scale = min(28 / h, 28 / w);

    newH = max(1, round(h * scale));
    newW = max(1, round(w * scale));

    resizedSmall = imresize(croppedImg, [newH newW]);

    % Create 32x32 blank canvas
    finalImg = zeros(32, 32);

    % Center the signature
    startRow = floor((32 - newH) / 2) + 1;
    startCol = floor((32 - newW) / 2) + 1;

    finalImg(startRow:startRow+newH-1, ...
             startCol:startCol+newW-1) = resizedSmall;

    %% Convert to feature vector

    featureVector = double(finalImg(:))';

    X(numGenuine+i,:) = featureVector;

    %% Forged = 0

    Y(numGenuine+i) = 0;

end

%% Display dataset information

fprintf("\n========================================\n");
fprintf("Improved Image Dataset Created\n");
fprintf("========================================\n");

fprintf("Total samples: %d\n", size(X,1));
fprintf("Total input features: %d\n", size(X,2));

%% Display first few features

disp("First 5 feature vectors:");

disp(X(1:5,1:20));

fprintf("Showing first 20 pixels only.\n");

%% Display labels

disp("First 10 labels:");

disp(Y(1:10));

%% Save dataset

save("signatureFeatures.mat", "X", "Y");

fprintf("\nDataset saved as signatureFeatures.mat\n");

%% Display processed signature

sampleImage = reshape(X(1,:), imageSize);

figure;

imshow(sampleImage);

title("Improved 32 x 32 Signature");