% gazescan_lr.m

%% ------------------------------------------------------------------------
% Do logistic regression learning on gaze set in ./output
% Directory contents are created by running gazescan.exe
% Files we expect:  
%   ./output/Features.png - The captured features
%   ./output/Coords.png - The coordinates that were being looked at

%% Initialization
clear ; close all; clc

%% ------------------------------------------------------------------------
% Load features and coords

% this_m_file_fullpath = mfilename('fullpath');
% idx = strfind(this_m_file_fullpath,'gazescan_lr');
this_m_file_directory = 'D:/projects/gazescan/';  % this_m_file_fullpath(1:idx-1);

%% ------------------------------------------------------------------------
% Features = imread([this_m_file_directory  'output/Features.png']);

%  read Labels (They're 0-8, so add 1)
%% Open the text file.
% fileID = fopen([this_m_file_directory  'output/Labels.txt'],'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
% Labels = textscan(fileID, '%d');

%% Close the text file.
% fclose(fileID);
% 
% Labels = 1 + Labels{:};

% m is training set size
% m = size(Features,1);
% if (m ~= size(Labels,1))
%     error('Number of features must match number of coords');
% end

% n is number of features
% n = size(Features,2);
%     
% % Convert Features .png image to normalized feature matrix X
% X = cast(Features, 'double');
% X = (X - mean(X, 1)) ./ std(X, 1);
%%
num_labels = 9;

%% Add bias column to X
% X = [ ones(size(X,1),1) X ];

%% Divide the data into training data and validation data
% train_validation_ratio = 3; % twice as much training data as validation data
% validation_row = m - m/(1 + train_validation_ratio);
% Xval = X(validation_row:end, :);
% yval = Labels(validation_row:end);
% 
% Xtrain = X(1:validation_row-1, :);
% ytrain = Labels(1:validation_row-1);
% 
%% Train the classifier with zero lambda

% lambda = 10.1; 
% all_theta = oneVsAll(Xtrain, ytrain, num_labels, lambda);
% 
% pred = predictOneVsAll(all_theta, Xval);
% 
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yval)) * 100);


%% Train a CNN with the images
% Create the data store


% CUDA fix
try
    nnet.internal.cnngpu.reluForward(1);
    catch ME
end

image_size = [32 32 1];
imds = imageDatastore([this_m_file_directory  'output/img'], 'IncludeSubfolders', 1, 'LabelSource', 'foldernames' );
% imds.Labels = categorical(cast(Labels,'uint8'));

label_counts = countEachLabel(imds);

numTrainingFiles = min(table2array(label_counts(:,2)));

fprintf('\nTraining with: %d training rows\n', numTrainingFiles);



[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles * 0.7, 'randomize');

layers = [ 
    imageInputLayer(image_size) 
    convolution2dLayer(5,20) 
    reluLayer 
    maxPooling2dLayer(2,'Stride',2) 
%    fullyConnectedLayer(num_labels) 
    fullyConnectedLayer(num_labels) 
    fullyConnectedLayer(num_labels) 
    softmaxLayer 
    classificationLayer
    ];
% options = trainingOptions('sgdm', 'MaxEpochs',100,'InitialLearnRate',1e-4, 'Verbose',false, 'Plots','training-progress');
options = trainingOptions('sgdm', 'MaxEpochs',200,'InitialLearnRate',1e-4, 'Verbose',false, 'Plots','training-progress');

%% Start the training
net = trainNetwork(imdsTrain,layers,options);

%% Validation
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

%% Export 
% exportONNXNetwork(net, 'gazescan.onnx');