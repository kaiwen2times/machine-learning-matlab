%% From Problem 4 of Hwk #5:
clear; close all; clc
 
% Load Training Data- Andrew Ng Machine Learning MOOC
addpath('../hw4');
load('ex3data1.mat'); % training data stored in arrays X, y (this data was given in hwk #4)
n = size(X, 1);
num_labels =  length(unique(y));          % 10 labels, from 1 to 10   (note  "0" is mapped to label 10)
 
% Randomly select 100 data points to display
rng(2000);  %random number generator seed
rand_indices = randperm(n);
sel = X(rand_indices(1:100), :);
 
%displayData(sel);
%print -djpeg95 hwk4_4.jpg
 
Xdata =X;
 
%ClassificationTree
tree = ClassificationTree.fit(Xdata,y);
maxprune = max(tree.PruneList);
treePrune = prune(tree,'level',maxprune-3);
view(treePrune,'mode','graph');
pred = predict(tree,Xdata);
subplot(2,1,1);
hold off; plot(y,'g','linewidth',2); hold on; plot(pred,'b','linewidth',2);
subplot(2,1,2);
plot(y-pred,'k','linewidth',3);
mse = (1/length(y))*sum((y-pred).^2);  %0.7964
fprintf('MSE for regular classification tree: %f \n ',mse);
 
%BaggedTree
rng(2000);  %random number generator seed
t = ClassificationTree.template('MinLeaf',1);
bagtree = fitensemble(Xdata,y,'Bag',10,t,'type','classification');
ypred = predict(bagtree,Xdata);  %really should test with a test set here
figure
subplot(2,1,1);
hold off; plot(y,'g','linewidth',2); hold on; plot(ypred,'b','linewidth',2);
subplot(2,1,2);
plot(y-ypred,'k','linewidth',3);
msebag = (1/length(y))*sum((y-ypred).^2);  %0.0884
fprintf('MSE for bagged classification tree: %f \n ',msebag);


%% Concrete challenge example

clear; close all; clc
Data = dlmread('Concrete_Data.csv', ',', 2, 0);  % 1029 samples, x [8 features , 1 target value]
% The 9 columns represent:
labels = {'Cement (component 1)(kg in a m^3 mixture)',
'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
'Fly Ash (component 3)(kg in a m^3 mixture)',
'Water  (component 4)(kg in a m^3 mixture)',
'Superplasticizer (component 5)(kg in a m^3 mixture)',
'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
'Fine Aggregate (component 7)(kg in a m^3 mixture)',
'Age (day)',
'Concrete compressive strength(MPa, megapascals)'}; %this last one is the target value
XdataTrain = Data(1:500,1:8);
yTrain = Data(1:500,9);

XdataTest = Data(501:1000,1:8);
yTest = Data(501:1000,9);

% Use any matlab built-in classification tree or regression tree function to get the 
% MSE below 225 and get 5 points on Exam #2.   
% (Can use pruning, boosting, bagging, etc. )  
% Need to use the exact train and test splits as above.  
% Need to use the exact MSE calculation as above

%You can't use the matlab functions templateTree, or fitctree as they unfairly
%advantage folks with newer matlab releases
% Replace:
%     t = templateTree( );
% with:
%    t = ClassificationTree.template( );
% or
%    t = RegressionTree.template( );
%
%Replace:
%   tree = fitctree( );
% with:
%   tree = ClassificationTree.fit( );
% or
%    tree = RegressionTree.fit( );

% For example: 
tree = RegressionTree.fit(XdataTrain,yTrain);
%maxprune = max(tree.PruneList);
%prune(tree,'level',maxprune-3);
% view(treePrune,'mode','graph');
pred = predict(tree,XdataTest);
mse = (1/length(yTest))*sum((yTest-pred).^2);
fprintf('MSE for  classification tree: %f \n ',mse);
% MSE for  classification tree: 328.045371


rng(2000);  %random number generator seed
t = RegressionTree.template('MinLeaf',1);
bagtree = fitensemble(XdataTrain,yTrain,'Bag',7,t,'type','regression');
ypred = predict(bagtree,XdataTest);
msebagclass = (1/length(yTest))*sum((yTest-ypred).^2);
fprintf('MSE for  bagged classficiation  tree: %f \n ',msebagclass);
% MSE for  bagged classficiation  tree: 544.878286 


treee = RegressionTree.template('MaxNumSplits',7,'MinLeaf',1);
boostTree = fitensemble(XdataTrain,yTrain,'LSBoost',10,treee,'type','regression');
ypred = predict(boostTree,XdataTest);
msebagclass = (1/length(yTest))*sum((yTest-ypred).^2);
fprintf('MSE for  bootst classficiation  tree: %f \n ',msebagclass);

%%


