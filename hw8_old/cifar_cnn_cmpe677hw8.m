function [net, info] = cifar_cnn_cmpe677hw8(network_arch)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
% Adapted from https://github.com/vlfeat/matconvnet/blob/master/examples/cnn_cifar.m
% Shagan Sah, Prof. Ray Ptucha, RIT 2015

filepath = 'C:\Users\ram_1\OneDrive\Documents\MATLAB\matconvnet' ;
run(fullfile(filepath,'matlab', 'vl_setupnn.m')) ;

rng(1000);
opts.train.learningRate = 0.05 ;
opts.train.weightDecay = 0.0001 ;

opts.expDir = fullfile('data', 'cifar') ;
% opts.expDir = fullfile('examples','data', 'cifar') ;

opts.train.numEpochs = 1;
opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.continue = false ;
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train.gpus = [] ;
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
% Prepare data and model
% --------------------------------------------------------------------
net = network_arch; %funtion call to initialize CNN arch. (cifart_cnn_initialize_sample)


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

[net, info] = cnn_train(net, imdb, @getBatch,opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
    labels{fi} = fd.labels' + 1; % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);
% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
    z = reshape(data,[],60000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
    z = reshape(data,[],60000) ;
    W = z(:,set == 1)*z(:,set == 1)'/60000 ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;