%% STEP 8-13: Improved Neural Network
% Intelligent Signature Verification and Forgery Detection System

clc;
clear;

%% Load Dataset

load("signatureFeatures.mat");

fprintf("Total samples: %d\n", size(X,1));
fprintf("Total features: %d\n", size(X,2));

%% Dataset Split

rng(42);

numSamples = size(X,1);

randomOrder = randperm(numSamples);

numTrain = round(0.70 * numSamples);

numValidation = round(0.15 * numSamples);

trainIndex = randomOrder(1:numTrain);

validationIndex = randomOrder( ...
    numTrain + 1 : ...
    numTrain + numValidation);

testIndex = randomOrder( ...
    numTrain + numValidation + 1 : end);

%% Create datasets

X_train = X(trainIndex,:);
Y_train = Y(trainIndex);

X_validation = X(validationIndex,:);
Y_validation = Y(validationIndex);

X_test = X(testIndex,:);
Y_test = Y(testIndex);

fprintf("\nDataset Split:\n");

fprintf("Training samples: %d\n", size(X_train,1));

fprintf("Validation samples: %d\n", ...
    size(X_validation,1));

fprintf("Testing samples: %d\n", ...
    size(X_test,1));

fprintf("\nTraining Genuine: %d\n", ...
    sum(Y_train == 1));

fprintf("Training Forged: %d\n", ...
    sum(Y_train == 0));

fprintf("Validation Genuine: %d\n", ...
    sum(Y_validation == 1));

fprintf("Validation Forged: %d\n", ...
    sum(Y_validation == 0));

fprintf("Testing Genuine: %d\n", ...
    sum(Y_test == 1));

fprintf("Testing Forged: %d\n", ...
    sum(Y_test == 0));

%% Normalize Features

mu = mean(X_train, 1);

sigma = std(X_train, 0, 1);

sigma(sigma == 0) = 1;

X_train_norm = ...
    (X_train - mu) ./ sigma;

X_validation_norm = ...
    (X_validation - mu) ./ sigma;

X_test_norm = ...
    (X_test - mu) ./ sigma;

fprintf("\nFeature normalization completed.\n");

fprintf("Normalized training data size: ");

disp(size(X_train_norm));

fprintf("Normalized validation data size: ");

disp(size(X_validation_norm));

fprintf("Normalized testing data size: ");

disp(size(X_test_norm));

%% Neural Network Architecture

inputSize = 1024;

hiddenSize1 = 64;

hiddenSize2 = 32;

outputSize = 1;

%% Training Parameters

learningRate = 0.001;

epochs = 1500;

%% Initialize Weights

rng(42);

% Input -> Hidden Layer 1

W1 = randn(inputSize, hiddenSize1) ...
    * sqrt(2 / inputSize);

b1 = zeros(1, hiddenSize1);

% Hidden Layer 1 -> Hidden Layer 2

W2 = randn(hiddenSize1, hiddenSize2) ...
    * sqrt(2 / hiddenSize1);

b2 = zeros(1, hiddenSize2);

% Hidden Layer 2 -> Output

W3 = randn(hiddenSize2, outputSize) ...
    * sqrt(2 / hiddenSize2);

b3 = zeros(1, outputSize);

%% Prepare Data

Xtr = X_train_norm;

Ytr = Y_train;

Xval = X_validation_norm;

Yval = Y_validation;

Xte = X_test_norm;

Yte = Y_test;

numTrainingSamples = size(Xtr,1);

%% ReLU Function

relu = @(z) max(0,z);

%% Training History

trainingLoss = zeros(epochs,1);

validationAccuracy = zeros(epochs,1);

%% Best Model Tracking

bestValidationAccuracy = 0;

bestW1 = W1;
bestb1 = b1;

bestW2 = W2;
bestb2 = b2;

bestW3 = W3;
bestb3 = b3;

%% TRAIN NETWORK

fprintf("\nTraining Improved Neural Network...\n");

for epoch = 1:epochs

    %% Forward Propagation

    Z1 = Xtr * W1 + b1;

    A1 = relu(Z1);

    Z2 = A1 * W2 + b2;

    A2 = relu(Z2);

    Z3 = A2 * W3 + b3;

    A3 = 1 ./ (1 + exp(-Z3));

    %% Prevent extreme values

    A3 = min(max(A3, eps), 1-eps);

    %% Binary Cross Entropy Loss

    loss = -mean( ...
        Ytr .* log(A3) + ...
        (1-Ytr) .* log(1-A3) ...
        );

    trainingLoss(epoch) = loss;

    %% Backpropagation

    dZ3 = A3 - Ytr;

    dW3 = ...
        (A2' * dZ3) / numTrainingSamples;

    db3 = mean(dZ3,1);

    dA2 = dZ3 * W3';

    dZ2 = dA2 .* (Z2 > 0);

    dW2 = ...
        (A1' * dZ2) / numTrainingSamples;

    db2 = mean(dZ2,1);

    dA1 = dZ2 * W2';

    dZ1 = dA1 .* (Z1 > 0);

    dW1 = ...
        (Xtr' * dZ1) / numTrainingSamples;

    db1 = mean(dZ1,1);

    %% Update Weights

    W3 = W3 - learningRate * dW3;

    b3 = b3 - learningRate * db3;

    W2 = W2 - learningRate * dW2;

    b2 = b2 - learningRate * db2;

    W1 = W1 - learningRate * dW1;

    b1 = b1 - learningRate * db1;

    %% Validation

    Z1_val = Xval * W1 + b1;

    A1_val = relu(Z1_val);

    Z2_val = A1_val * W2 + b2;

    A2_val = relu(Z2_val);

    Z3_val = A2_val * W3 + b3;

    A3_val = ...
        1 ./ (1 + exp(-Z3_val));

    %% Validation Accuracy at 0.5

    validationPredicted = ...
        A3_val >= 0.5;

    currentValidationAccuracy = ...
        mean(validationPredicted == Yval) * 100;

    validationAccuracy(epoch) = ...
        currentValidationAccuracy;

    %% Save Best Network

    if currentValidationAccuracy > ...
            bestValidationAccuracy

        bestValidationAccuracy = ...
            currentValidationAccuracy;

        bestW1 = W1;
        bestb1 = b1;

        bestW2 = W2;
        bestb2 = b2;

        bestW3 = W3;
        bestb3 = b3;

    end

    %% Display Progress

    if mod(epoch,100) == 0

        fprintf( ...
            "Epoch %d/%d | Loss: %.4f | Validation Accuracy: %.2f%%\n", ...
            epoch, epochs, loss, ...
            currentValidationAccuracy);

    end

end

fprintf("\nTraining completed!\n");

fprintf("Best Validation Accuracy: %.2f%%\n", ...
    bestValidationAccuracy);

%% Restore Best Network

W1 = bestW1;
b1 = bestb1;

W2 = bestW2;
b2 = bestb2;

W3 = bestW3;
b3 = bestb3;

%% Find Best Decision Threshold

fprintf("\nFinding optimal decision threshold...\n");

% Forward propagation on validation set

Z1_val = Xval * W1 + b1;

A1_val = relu(Z1_val);

Z2_val = A1_val * W2 + b2;

A2_val = relu(Z2_val);

Z3_val = A2_val * W3 + b3;

validationScores = ...
    1 ./ (1 + exp(-Z3_val));

thresholds = 0.30:0.01:0.70;

bestThreshold = 0.50;

bestThresholdAccuracy = 0;

for t = thresholds

    tempPrediction = ...
        validationScores >= t;

    tempAccuracy = ...
        mean(tempPrediction == Yval) * 100;

    if tempAccuracy > ...
            bestThresholdAccuracy

        bestThresholdAccuracy = ...
            tempAccuracy;

        bestThreshold = t;

    end

end

fprintf("Best Threshold: %.2f\n", ...
    bestThreshold);

fprintf("Validation Accuracy at Best Threshold: %.2f%%\n", ...
    bestThresholdAccuracy);

%% TEST NETWORK

Z1_test = Xte * W1 + b1;

A1_test = relu(Z1_test);

Z2_test = A1_test * W2 + b2;

A2_test = relu(Z2_test);

Z3_test = A2_test * W3 + b3;

testScores = ...
    1 ./ (1 + exp(-Z3_test));

%% Apply Best Threshold

testPredicted = ...
    testScores >= bestThreshold;

%% Test Accuracy

testAccuracy = ...
    mean(testPredicted == Yte) * 100;

fprintf("\n====================================\n");

fprintf("FINAL TEST ACCURACY: %.2f%%\n", ...
    testAccuracy);

fprintf("====================================\n");

%% CONFUSION MATRIX

actual = Yte(:);

predicted = double(testPredicted(:));

TP = sum((actual == 1) & ...
         (predicted == 1));

TN = sum((actual == 0) & ...
         (predicted == 0));

FP = sum((actual == 0) & ...
         (predicted == 1));

FN = sum((actual == 1) & ...
         (predicted == 0));

%% Metrics

accuracy = ...
    (TP + TN) / ...
    (TP + TN + FP + FN) * 100;

precision = ...
    TP / (TP + FP + eps) * 100;

recall = ...
    TP / (TP + FN + eps) * 100;

F1 = ...
    2 * (precision * recall) / ...
    (precision + recall + eps);

%% Display Performance

fprintf("\n========================================\n");

fprintf("IMPROVED MODEL PERFORMANCE\n");

fprintf("========================================\n");

fprintf("True Positives  : %d\n", TP);

fprintf("True Negatives  : %d\n", TN);

fprintf("False Positives : %d\n", FP);

fprintf("False Negatives : %d\n", FN);

fprintf("----------------------------------------\n");

fprintf("Accuracy  : %.2f%%\n", accuracy);

fprintf("Precision : %.2f%%\n", precision);

fprintf("Recall    : %.2f%%\n", recall);

fprintf("F1-Score  : %.2f%%\n", F1);

fprintf("Threshold : %.2f\n", bestThreshold);

fprintf("========================================\n");

%% Confusion Matrix

confusionMatrix = [TN FP;
                   FN TP];

figure;

imagesc(confusionMatrix);

colorbar;

title("Improved Model - Confusion Matrix");

xlabel("Predicted Class");

ylabel("Actual Class");

xticks([1 2]);

xticklabels({"Forged","Genuine"});

yticks([1 2]);

yticklabels({"Forged","Genuine"});

for i = 1:2

    for j = 1:2

        text(j,i, ...
            num2str(confusionMatrix(i,j)), ...
            'HorizontalAlignment','center', ...
            'FontSize',14);

    end

end

%% Training Loss

figure;

plot(trainingLoss, ...
    'LineWidth',1.5);

xlabel("Epoch");

ylabel("Loss");

title("Improved Neural Network Training Loss");

grid on;

%% Validation Accuracy

figure;

plot(validationAccuracy, ...
    'LineWidth',1.5);

xlabel("Epoch");

ylabel("Accuracy (%)");

title("Improved Neural Network Validation Accuracy");

grid on;

%% Save Improved Network

save("signatureNeuralNetwork.mat", ...
    "W1","b1","W2","b2","W3","b3", ...
    "mu","sigma","bestThreshold");

fprintf("\nImproved Neural Network saved as signatureNeuralNetwork.mat\n");