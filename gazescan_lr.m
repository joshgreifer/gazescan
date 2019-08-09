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

PREPROCESS_FILES = 0;
RUN_GAZESCAN = 0;

% CUDA fix
try
    nnet.internal.cnngpu.reluForward(1);
    catch ME
end

image_size = [48 96 1];

% imds = imageDatastore([this_m_file_directory  'output/img'], 'IncludeSubfolders', 1,'LabelSource', 'foldernames', ...
% 'ReadFcn', @CustomImgReaderConv2WithCalibration ...
% );

imds = imageDatastore([this_m_file_directory  'output/img'], 'IncludeSubfolders', 1,'LabelSource', 'foldernames');

% 'ReadFcn', @CustomImgReaderAbsDiffWithCalibration ...

% Running net using image datastore with CustomReader was
% too slow, so do it in a batch here
if PREPROCESS_FILES == 1
    [~, ~] = mkdir([this_m_file_directory  'preprocessed/img']);
    num_files = size(imds.Files);
    fprintf('Preprocessing file, patience...');
    for i=1:num_files
        src_file_path = imds.Files{i};
         dest_file_path = strrep(src_file_path, 'output', 'preprocessed');
        if exist(dest_file_path, 'file') ~= 2
             k = strfind(dest_file_path, 'Img');
            dest_dir = dest_file_path(1:k-1);
            [~, ~] = mkdir(dest_dir);
            img = CustomImgReaderAbsDiffWithCalibration(src_file_path);
            imwrite(img, dest_file_path);
        end
        % s
    end
    fprintf('Preprocessing done.  Files are in %s ', [this_m_file_directory  'preprocessed/img']);
end
% imds = imageDatastore([this_m_file_directory  'preprocessed/img'], 'IncludeSubfolders', 1, 'LabelSource', 'foldernames');
% 'ReadFcn', @CustomImgReaderAbsDiffWithCalibration ...



label_counts = countEachLabel(imds);

numTrainingFiles = min(table2array(label_counts(:,2)));

fprintf('\nTraining with: %d training rows\n', numTrainingFiles);



[imdsTrain,imdsTest] = splitEachLabel(imds,0.5, 'randomize');

layers = [ 
    imageInputLayer(image_size, 'name', 'img_input', 'Normalization', 'none') 
    
    convolution2dLayer(3,16)
    batchNormalizationLayer
    reluLayer 
%     
%     averagePooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,8)
%     batchNormalizationLayer
%     reluLayer 
%     
%     averagePooling2dLayer(2,'Stride',2) 
%     
%     
%     convolution2dLayer(3,32)
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(3,32)
%     batchNormalizationLayer
%     reluLayer
%     
%      fullyConnectedLayer(32) 
%      fullyConnectedLayer(10) 
    fullyConnectedLayer(num_labels) 
    softmaxLayer 
    classificationLayer
    ];

load 'designed_net.mat';

% options = trainingOptions('sgdm', 'MaxEpochs',100,'InitialLearnRate',1e-4, 'Verbose',false, 'Plots','training-progress');
options = trainingOptions('sgdm', 'MaxEpochs',500,'InitialLearnRate',1e-4, 'Verbose',false, 'Plots','training-progress','ValidationData', imdsTest);

%% Start the training
imdsTrainAugmented = augmentedImageDatastore(image_size,imdsTrain);

% net = trainNetwork(imdsTrain,designed_net,options);
net = trainNetwork(imdsTrainAugmented,layers,options);

%% Validation
[YPred, yScores] = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

%% Export 
exportONNXNetwork(net, 'gazescan.onnx');


results_matlab = [imdsTest.Files num2cell(yScores)];
results_gazescan = zeros(length(imdsTest.Files), height(label_counts));
%% Run in executable
if RUN_GAZESCAN
    for i=1:size(results_matlab)
        filename = results_matlab{i,1};

        cmd = [ this_m_file_directory 'x64/Release/gazescan.exe ' filename ];
        [status, output] = system(cmd);
        idx = strfind(output,'[');
        r = sscanf(output(idx+1:end-2), '%f, %f, %f, %f, %f, %f, %f, %f, %f');
        results_gazescan(i, :) = r';
    end
end
