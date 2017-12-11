%function [convnet, info] = cifar_cnn_cmpe677hw8_nnt(network_arch)
% CNN_CIFAR   Demonstrates  CNN using Neural Net Toolbox on CIFAR-10
% Adapted from https://www.mathworks.com/help/vision/examples/object-detection-using-deep-learning.html
% Prof. Ray Ptucha, RIT 2017
clear;clc;
rng(1000);
opts.train.learningRate = 0.001 ;

opts.expDir = fullfile('data', 'cifar') ;

opts.train.numEpochs = 1;
opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 250 ;
opts.train.continue = false ;
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train.gpus = [] ;
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
% Prepare data and model
% --------------------------------------------------------------------


if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getCifarImdb(opts) ;
    if (~exist(opts.dataDir,'dir'))  %rwp update
        mkdir(opts.dataDir) ;
    end
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
% Train
% --------------------------------------------------------------------
 %function call to initialize CNN arch. (cifart_cnn_initialize_sample)

[layers, options] = cifar_cnn_architecture1(opts);

if exist('cifar10Net1.mat', 'file')
    load cifar10Net1;
else
%Train the network.
[cifar10Net1, info] = trainNetwork( imdb.images.data_train,...
                            imdb.images.labels_train_gt,...
                            layers, options);
                   
save cifar10Net1;
end


%Run the trained network on the test set that was not used to train the network and predict the image labels (digits).
YTest = classify(cifar10Net1,imdb.images.data_test);
%Calculate the accuracy.
accuracy = sum(YTest' == imdb.images.labels_test_gt)/numel(imdb.images.labels_test_gt)



