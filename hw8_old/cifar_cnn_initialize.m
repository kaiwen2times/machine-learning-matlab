function net = cifar_cnn_initialize()

rng(1000);

lr = [.1 2] ; % learning rate for weights and bias

% Define network
net.layers = {} ;

% Layer 1
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.01*randn(7,7,3,32, 'single'), zeros(1, 32, 'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 3) ;
% Layer 2
net.layers{end+1} = struct('type', 'relu') ;
% Layer 3
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]);
% Layer 4
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*randn(5,5,32,32, 'single'), zeros(1,32,'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 2) ;
% Layer 5
net.layers{end+1} = struct('type', 'relu') ;
% Layer 6
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]);
% Layer 7
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*randn(3,3,32,64, 'single'), zeros(1,64,'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 1) ;
% Layer 8
net.layers{end+1} = struct('type', 'relu') ;
% Layer 9
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]);
% Layer 10
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*randn(4,4,32,128, 'single'), zeros(1,128,'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;
% Layer 11
% Layer 12
net.layers{end+1} = struct('type', 'relu') ;
% Layer 13
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*randn(1,1,128, 10, 'single'), zeros(1,10,'single')}}, ...
    'learningRate', .1*lr, ...
    'stride', 1, ...
    'pad', 0) ;
% Loss layer no. 14
net.layers{end+1} = struct('type', 'softmaxloss') ;
