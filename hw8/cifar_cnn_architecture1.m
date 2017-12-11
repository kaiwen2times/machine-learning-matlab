function [layers, options] = cifar_cnn_architecture(opts)

% Set the network training options
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', opts.train.learningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', opts.train.numEpochs, ...
    'MiniBatchSize', opts.train.batchSize, ...
    'Verbose', true, 'OutputFcn',@plotTrainingAccuracy); 

% Create the image input layer for 32x32x3 CIFAR-10 images
inputLayer = imageInputLayer([32 32 3]);

%%
% Next, define the middle layers of the network. The middle layers are made
% up of repeated blocks of convolutional, ReLU (rectified linear units),
% and pooling layers. These 3 layers form the core building blocks of
% convolutional neural networks. The convolutional layers define sets of
% filter weights, which are updated during network training. The ReLU layer
% adds non-linearity to the network, which allow the network to approximate
% non-linear functions that map image pixels to the semantic content of the
% image. The pooling layers downsample data as it flows through the
% network. In a network with lots of layers, pooling layers should be used
% sparingly to avoid downsampling the data too early in the network.

% Convolutional layer parameters
filterSize7 = [7 7];
filterSize5 = [5 5];
filterSize3 = [3 3];
numFilters = 32;

middleLayers = [
convolution2dLayer(filterSize7, numFilters, 'Padding', 3)
reluLayer()
maxPooling2dLayer(3, 'Stride', 2)


convolution2dLayer(filterSize5, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize3, 2 * numFilters, 'Padding', 1)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

]


finalLayers = [
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(10)
softmaxLayer
classificationLayer
]

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

layers(2).Weights = 0.0001 * randn([filterSize7 3 numFilters]);


