clear
clc
close all
%==========================================================================
% question 1
%==========================================================================
figure
dataX = [1, 2, 3, 2, 5];
dataY = [5, 6, 10, 1, 8.5];
% linear classifier
linX = 0:0.1:5.5;
linY = linX.*2 + 0.5;
subplot(1,2,1)
plot(linX,linY,'b--','MarkerSize',10,'LineWidth',3)
hold on
scatter(dataX(1:3),dataY(1:3),60,'^')
scatter(dataX(4:5),dataY(4:5),60,'v')
axis([0 10 0 11])
title('Linear Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on
% decision tree classifier
dtX = 0:0.1:10;
dtY = ones(length(dtX)).*4;
subplot(1,2,2)
plot(dtX,dtY,'b--','MarkerSize',10,'LineWidth',3)
hold on
scatter(dataX(1:3),dataY(1:3),60,'^')
scatter(dataX(4:5),dataY(4:5),60,'v')
axis([0 10 0 11])
title('Descision Tree Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on

% Given n points, n is the maximum number of tree nodes to do perfect classification


%==========================================================================
% question 2
%==========================================================================
figure
dataX = [1, 2, 3, 5, 6];
dataY = [10, 7, 4, 7, 8.5];
% linear classifier
linX = 0:0.1:5.5;
linY = linX.*2 + 0.5;
subplot(1,2,1)
plot(linX,linY,'b--','MarkerSize',10,'LineWidth',3)
hold on
scatter(dataX(1:3),dataY(1:3),60,'^')
scatter(dataX(4:5),dataY(4:5),60,'v')
axis([0 10 0 11])
title('Linear Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on
% decision tree classifier
dtY = 0:0.1:11;
dtX = ones(length(dtY)).*4;
subplot(1,2,2)
plot(dtX,dtY,'b--','MarkerSize',10,'LineWidth',3)
hold on
scatter(dataX(1:3),dataY(1:3),60,'^')
scatter(dataX(4:5),dataY(4:5),60,'v')
axis([0 10 0 11])
title('Descision Tree Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on


%==========================================================================
% question 3
%==========================================================================





%==========================================================================
% question 4
%==========================================================================
clear
% Load Training Data- Andrew Ng Machine Learning MOOC
load('ex3data1.mat'); % training data stored in arrays X, y (this data was given in hwk #4)
n = size(X, 1);
num_labels = length(unique(y)); % 10 labels, from 1 to 10 (note "0" is mapped to label 10)

% Randomly select 100 data points to display
rng(2000); %random number generator seed
rand_indices = randperm(n);
sel = X(rand_indices(1:100), :);

%displayData(sel);
%print -djpeg95 hwk4_4.jpg

Xdata = X;
figure

%ClassificationTree
tree = ClassificationTree.fit(Xdata,y);
maxprune = max(tree.PruneList);
treePrune = prune(tree,'level',maxprune-3);
view(treePrune,'mode','graph');
pred = predict(tree,Xdata);
subplot(2,1,1);
hold off;
plot(y,'g','linewidth',2); hold on; plot(pred,'b','linewidth',2);
subplot(2,1,2);
plot(y-pred,'k','linewidth',3);
mse = (1/length(y))*sum((y-pred).^2); %0.7964
fprintf('MSE for regular classification tree: %f \n ',mse);

%BaggedTree
rng(2000); %random number generator seed
t = ClassificationTree.template('MinLeaf',1);
bagtree = fitensemble(Xdata,y,'Bag',10,t,'type','classification');
ypred = predict(bagtree,Xdata); %really should test with a test set here
figure
subplot(2,1,1);
hold off;
plot(y,'g','linewidth',2); hold on; plot(ypred,'b','linewidth',2);
subplot(2,1,2);
plot(y-ypred,'k','linewidth',3);
msebag = (1/length(y))*sum((y-ypred).^2); %0.0884
fprintf('MSE for bagged classification tree: %f \n ',msebag);