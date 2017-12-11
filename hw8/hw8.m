clear;
clc;
close all;
%==========================================================================
% question 1
%==========================================================================
% filter 3x3x3
% output 16x16x96


%==========================================================================
% question 2
%==========================================================================
% output 16x16x96


%==========================================================================
% question 3a
%==========================================================================
% a. How many activation layers are present in this network?
% 2, one ReLU, one softmax
% b. How many Pooling layers are present in this network?
% 1
% c. What are the filter dimensions and number of filters in the first Convolutional layer?
% 5x5x20
% d. What is the size of the image after applying the Convolutional layer?
% 20x20x20
% e. What is the batch size we are using in this model?
% 10
% f. What is the learning rate value used in this model?
% 0.001


%==========================================================================
% question 8
%==========================================================================
%% Dim Reduction 
close all ; clear all;
%Download hwk8files_forMycourses.zip and place them into an appropriate
%hwk8 directory.  Update the two paths below for your machine
cd C:\Users\Mellow\Downloads\machine-intelligence\hw8
addpath C:\Users\Mellow\Downloads\machine-intelligence\hw6\libsvm-3.18\windows
 
%  This cell will demonstrate a comparison of unsupervised PCA vs.
%  Supervised methods such as supervised LPP
load LPP_example_data.mat
% 1072 faces from Kohn-Canade (they have been cropped and resampled)
% GT- ground truth:
%   100: angry
%   200: happy
%   300: neutral
%   400: sad
%   500: surprised
% allfaces20- 26x20 images
 
[numfaces,height,width] = size(allfaces20);
allfacesMean128_Std100=allfaces20;  % this allocates memory, makes it faster
faceu = uint8(zeros(height,width));
for i=1:numfaces
    faceu(:) = uint8(allfaces20(i,:,:));
    %imshow(faceu);pause   %use this line if you want to display the faces
    
    %fix mean to be 128, std dev to be 100
    allfacesMean128_Std100(i,:,:) = uint8((double(faceu)-mean(reshape(double(faceu),height*width,1)))*100/(std(reshape(double(faceu),height*width,1)+0.000001))+ 128);
end
imshow(faceu); %display the last face
     
% Now we vectorize our data:  numSamples*numDim; numDim=26*20=520
allfacesVnorm = zeros(numfaces,height*width);
allfacesVnorm(:) = allfacesMean128_Std100;
 
fea_train = allfacesVnorm;  %fea_train = <num_train_samples> x <D>

%
% First do PCA Analysis
% You will have to set up training and test sets using some sort of k-fold
% cross validataion.  This is skipped here for simplicity
%
cov_mat_fea_train = cov(fea_train);
% eigvector = <DxD>, eigvalue = <DxD>
[PCA_eigvector, PCA_eigvalue] = eig(cov_mat_fea_train);
% eigvalue = <Dx1>
PCA_eigvalue = diag(PCA_eigvalue);
 
%sort so most important is top to bottom, left to right
PCA_eigvalue = flipud(PCA_eigvalue);
PCA_eigvector = fliplr(PCA_eigvector);
 
% Now apply PCA eig matrix on data and look at output
PCA_output = fea_train*PCA_eigvector;   %[nxD]*[DxD]  --> [nxD] 
 
%
% Now call LPP code in supervised mode
%
options.Metric = 'Euclidean';
options.NeighborMode = 'Supervised';
options.gnd = GT';          %gnd = 1x<num_train_samples>
options.bLDA = 1;
% W is the Laplacian matrix:
% In supervised mode, this is from ground truth
% In unsupervised mode, this is from neighbor distances
W = constructW(fea_train,options);  % W is <num_train_samples> x <num_train_samples>
options.PCARatio = 0.99;
%options.PCARatio = 1.0;
data=fea_train;
[LPP_eigvector, LPP_eigvalue] = lpp(W, options, fea_train); % eigvector = <Dxd>, eigvalue = <dx1>
 
% Now apply LPP eig matrix on data and look at output
LPP_output = fea_train*LPP_eigvector;   %[nxD]*[Dxd]  --> [nxd] top d dims  

%
% Plot output
%
numclasses = max(GT)/100;  %assumes 100, 200, 300, ... for each class
class_symbol{1}='ro';
class_symbol{2}='gs';
class_symbol{3}='bp';
class_symbol{4}='cd';
class_symbol{5}='m+';
 
method='PCA';
abc123 = PCA_output(:,1:3)';  %restrict to 3 dims
% Note: PCA needs more than 3 dimms to classify data
% Use plot(PCA_eigvalue) to see variance of each dim
[numdim,numsubjects] = size(abc123);
abc123_avg = zeros(numclasses,3);
classcount = zeros(numclasses,1);
figure
for k=1:numsubjects
    subj=zeros(1,3);
    class = GT(k)/100;
    %[i j k abc123(:,count)']
    str = class_symbol{class};
 
    plot3d(abc123(:,k)',str,'linewidth',2); %%this looks better if standalone plot
    hold on
    abc123_avg(class,:) = abc123_avg(class,:) + abc123(:,k)';
    classcount(class) = classcount(class)+1;
end
        
for i=1:numclasses
    abc123_avg(i,:) = abc123_avg(i,:)./ classcount(i);
end
    
xlabel(sprintf('%s: Dim 1',method));
ylabel(sprintf('%s: Dim 2',method));
zlabel(sprintf('%s: Dim 3',method));
title(sprintf('%s on Expression Data: r=angry; g=happy; b=neutral; c=sad; m=surprised',method));
grid on
% This will save plot to disk, png is best for vector graphics
% jpeg is best for images
% print -dpng sampleoutputPCA.png  
 
 
method='SLPP';
abc123 = LPP_output(:,1:3)';  %restrict to 3 dims
[numdim,numsubjects] = size(abc123);
abc123_avg = zeros(numclasses,3);
classcount = zeros(numclasses,1);
figure
for k=1:numsubjects
    subj=zeros(1,3);
    class = GT(k)/100;
    %[i j k abc123(:,count)']
    str = class_symbol{class};
 
    plot3d(abc123(:,k)',str,'linewidth',2); %%this looks better if standalone plot
    hold on
    abc123_avg(class,:) = abc123_avg(class,:) + abc123(:,k)';
    classcount(class) = classcount(class)+1;
end
        
for i=1:numclasses
    abc123_avg(i,:) = abc123_avg(i,:)./ classcount(i);
end
    
xlabel(sprintf('%s: Dim 1',method));
ylabel(sprintf('%s: Dim 2',method));
zlabel(sprintf('%s: Dim 3',method));
title(sprintf('%s on Expression Data: r=angry; g=happy; b=neutral; c=sad; m=surprised',method));
grid on
% print -dpng sampleoutputSLPP.png



%==========================================================================
% question 8a
%==========================================================================
%% SVM vs NNet
close all; clear all;
load ex4data1.mat;
cd C:\Users\Mellow\Downloads\machine-intelligence\hw8
addpath C:\Users\Mellow\Downloads\machine-intelligence\hw6\libsvm-3.18\windows

options.method = 'SVM';
options.numberOfFolds = 5;
options.useDR=1;
options.dim_reduction='SLPP';
options.SLPP_bLDA=0.7;
options.PCARatio=0.9;
[confusionMatrixSVM,accuracySVM] =  classify677_hwk8(X,y,options);

options.method = 'nnet';
options.nnet_hiddenLayerSize = 25;
options.numberOfFolds = 5;
options.useDR=1;
options.dim_reduction='SLPP';
options.SLPP_bLDA=0.7;
options.PCARatio=0.9;
[confusionMatrixNN,accuracyNN] =  classify677_hwk8(X,y,options);
