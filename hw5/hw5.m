clear;
clc;
close all;
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
print('cmpe677_hwk5_1','-dpng')

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
print('cmpe677_hwk5_2','-dpng')

%==========================================================================
% question 3
%==========================================================================





%==========================================================================
% question 4
%==========================================================================
clear;
% Load Training Data- Andrew Ng Machine Learning MOOC
load('ex3data1.mat'); % training data stored in arrays X, y
n = size(X, 1);
num_labels = length(unique(y)); % 10 labels, from 1 to 10 (note "0" is mapped to label 10)

% Randomly select 100 data points to display
rng(2000); %random number generator seed
rand_indices = randperm(n);
sel = X(rand_indices(1:100), :);

%displayData(sel);
%print -djpeg95 hwk4_4.jpg

Xdata = [ones(n, 1) X];
% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds = 5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);

method = 'ClassificationTree'

lambda = 0.1;
for i = 1:numberOfFolds
  TestIndex = find(CVindex == i);
  TrainIndex = find(CVindex ~= i);
  TrainDataCV = Xdata(TrainIndex,:);
  TrainDataGT =y(TrainIndex);
  TestDataCV = Xdata(TestIndex,:);
  TestDataGT = y(TestIndex);
  %
  %build the model using TrainDataCV and TrainDataGT
  %test the built model using TestDataCV
  %
  switch method
    case 'LogisticRegression'
      % for Logistic Regression, we need to solve for theta
      % Initialize fitting parameters
      all_theta = zeros(num_labels, size(Xdata, 2));
      for c=1:num_labels
        % Set Initial theta
        initial_theta = zeros(size(Xdata, 2), 1);
        % Set options for fminunc
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        % Run fmincg to obtain the optimal theta
        % This function will return theta and the cost
        [theta] = fmincg (@(t)(costFunctionLogisticRegression(t, TrainDataCV, (TrainDataGT == c), lambda)), ...
        initial_theta, options);
        all_theta(c,:) = theta;
      end
      % Using TestDataCV, compute testing set prediction using
      % the model created
      % for Logistic Regression, the model is theta
      % Insert code here to see how well theta works...
      all_pred = sigmoid(TestDataCV*all_theta');
      [maxVal,maxIndex] = max(all_pred,[],2);
      TestDataPred=maxIndex;
    case 'KNN'
      [idx, dist] = knnsearch(TrainDataCV,TestDataCV,'k',3);
      TestDataPred=mode([TrainDataGT(idx(:,1)) TrainDataGT(idx(:,2)) TrainDataGT(idx(:,3)) ]')';
    case 'ClassificationTree'
      tree = ClassificationTree.fit(TrainDataCV,TrainDataGT);
      maxprune = max(tree.PruneList);
      treePrune = prune(tree,'level',maxprune-3);
      view(treePrune,'mode','graph');
      TestDataPred = predict(tree,TestDataCV);
    case 'BaggedTree'
      rng(2000); %random number generator seed
      t = ClassificationTree.template('MinLeaf',1);
      bagtree = fitensemble(TrainDataCV,TrainDataGT,'Bag',10,t,'type','classification');
      TestDataPred = predict(bagtree,TestDataCV); %really should test with a test set here
    otherwise
      error('Unknown classification method')
  end
  predictionLabels(TestIndex,:) =double(TestDataPred);
end

confusionMatrix = confusionmat(y,predictionLabels);
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));

fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method, lambda,accuracy*100));
fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix);
for i=1:r
  for j=1:r
    fprintf('%6d ',confusionMatrix(i,j));
  end
  fprintf('\n');
end



%==========================================================================
% question 6a
%==========================================================================
clear
addpath('../hw4');
b = load('bupa.data');
X = b(:,1:end-1);
y = b(:,end);
method = 'LogisticRegression'
options.method = method;
options.numberOfFolds = 5;
options.lambda = 0.1;
[confusionMatrix,accuracy] = classifiy677_hwk5(X,y,options);


%==========================================================================
% question 6e
%==========================================================================
clear
b = load('bupa.data');
Xdata = b(:,1:end-1);
y = b(:,end);

%turn ground truth labels into {-1,+1}
yList = unique(y);
if yList(1) ~= -1
  y(y==yList(1))=-1;
  y(y==yList(2))= 1;
end
%form train and test sets
TrainXdata = Xdata(1:200,:);
TrainGT = y(1:200);
TestXdata = Xdata(201:end,:);
TestGT = y(201:end);

%number of features
adaboost_numFeatures=500;
[classifiers, errors, pred] = myAdaBoost(TrainXdata,TrainGT,adaboost_numFeatures,TestXdata,TestGT)
meanTrain = errors.train;
meanTest = errors.test;
meanEB = errors.eb;

figure
hold on
x = 1:1:adaboost_numFeatures;
plot(x, meanEB,'k:',x,meanTest,'r-',x,meanTrain,'b--','LineWidth',2);
legend('ErrorBound','TestErr','TrainErr','Location','Best');
xlabel 'iteration (number of Classifiers)'
ylabel 'error rates (50 trials)'
title 'AdaBoost Performance on Bupa'
print('cmpe677_hwk5_6','-dpng')



%==========================================================================
% question 7
%==========================================================================
clear
addpath('../hw4');
b = load('bupa.data');
X = b(:,1:end-1);
y = b(:,end);
options.method = 'Adaboost';
options.numberOfFolds = 5;
options.adaboost_numFeatures = 500;
[confusionMatrix,accuracy] = classifiy677_hwk5(X,y,options)