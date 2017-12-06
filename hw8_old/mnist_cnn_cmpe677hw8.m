function [net, info] = mnist_cnn_cmpe677hw8(network_arch)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
% Adapted from https://github.com/vlfeat/matconvnet/blob/master/examples/cnn_cifar.m
% Shagan Sah, Prof. Ray Ptucha, RIT 2015

filepath = 'C:\Users\ram_1\OneDrive\Documents\MATLAB\matconvnet' ;
run(fullfile(filepath,'matlab', 'vl_setupnn.m')) ;

rng(1000);
opts.train.learningRate = 0.001 ;
opts.train.numEpochs = 1;
opts.dataDir = fullfile('.\') ;
opts.imdbPath = fullfile(opts.dataDir, 'ex4data1_lmdb');
opts.train.batchSize = 10 ;
opts.train.continue = false ;
opts.train.gpus = [] ;
opts.train.expDir = opts.dataDir ;

% --------------------------------------------------------------------
% Prepare data and model
% --------------------------------------------------------------------
net = network_arch; %funtion call to initialize CNN arch. (mnist_cnn_initialize_sample)


if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getMnistImdb(opts) ;
    if (~exist(opts.dataDir,'dir'))
        mkdir(opts.dataDir) ;
    end
    save(opts.imdbPath, '-struct', 'imdb') ;
end


% --------------------------------------------------------------------
% Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch,opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted

load('ex4data1.mat');

set = [ones(1,4000) 3*ones(1,1000)];

idx = randperm(size(X,1));
X=X(idx,:);
y=y(idx);

data = permute(single(reshape(X,[],20,20,1)),[2 3 4 1]);
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = y' ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),[10,1:9],'uniformoutput',false) ;
