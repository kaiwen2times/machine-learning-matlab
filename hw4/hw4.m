%==========================================================================
% question 1
%==========================================================================
% Load Data (from Andrew Ng Machine Learning online MOOC)
% The first two columns contains the X values and the third column % contains the label (y).
clear;
close all;
data = load('ex2data2.txt'); %data is 118x3
X = data(:, [1, 2]);
y = data(:, 3);
index0 = find(y == 0);
index1 = find(y == 1);
figure
plot(X(index0,1),X(index0,2),'ro');
hold on
plot(X(index1,1),X(index1,2),'g+');
% Labels and Legend
xlabel('Microchip Test 1','fontsize',12);
ylabel('Microchip Test 2','fontsize',12);
legend('y = 0', 'y = 1');

degree=6; %degree of polynomial allowed
Xdata = mapFeature(X(:,1), X(:,2),degree);
% Initialize fitting parameters
initial_theta = zeros(size(Xdata, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionLogisticRegression_slow(initial_theta, Xdata, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost); %should be about 0.693

[cost2, grad2] = costFunctionLogisticRegression(initial_theta, Xdata, y, lambda);
fprintf('Cost at initial theta without for loop: %f\n', cost2); %should be about 0.693


%==========================================================================
% question 2
%==========================================================================
% Initialize fitting parameters
initial_theta = zeros(size(Xdata, 2), 1);
% Set regularization parameter lambda to 1 (you should vary this)
lambda = [0, 1, 10, 100];
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
threshold = 0.5;
% Optimize
figure
for index = 1:4
  [theta, J, exit_flag] = fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda(index))), initial_theta, options);
  subplot(2, 2, index)
  plotDecisionBoundary(theta, Xdata, y, degree)
  xlabel('Microchip Test 1','fontsize',10);
  ylabel('Microchip Test 2','fontsize',10);
  hypo = (sigmoid(Xdata * theta) >= threshold);
  result = (hypo == y);
  acc = sum(result)/length(y);
  str = strcat('\lambda=',num2str(lambda(index)),', Accuracy%=',num2str(acc));
  title(str,'fontsize',10)
end
print('cmpe677_hwk4_2_circle','-dpng')



%==========================================================================
% question 3
%==========================================================================
numberOfFolds=5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y,numberOfFolds);
method='LogisticRegression'
lambda=100;
predictionLabels = zeros(size(y));

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
      % Insert code here to solve for theta...
      % Using TestDataCV, compute testing set prediction using the model created
      % for Logistic Regression, the model is theta
      % Insert code here to see how well theta works...
      initial_theta = zeros(size(Xdata, 2), 1);
      options = optimset('GradObj', 'on', 'MaxIter', 400);
      theta = fminunc(@(t)(costFunctionLogisticRegression(t, TrainDataCV, TrainDataGT, lambda)), initial_theta, options);
      hypo = (sigmoid(TestDataCV * theta) >= threshold);
    case 'KNN'
      disp('KNN not implemented yet')
    otherwise
      error('Unknown classification method')
  end
  predictionLabels(TestIndex,:) = double(hypo);
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
% question 4, 5
%==========================================================================
clear
% Load Training Data- Andrew Ng Machine Learning MOOC
load('ex3data1.mat'); % training data stored in arrays X, y
n = size(X, 1);
num_labels = length(unique(y)); % 10 labels, from 1 to 10 (note "0" is mapped to label 10)
% Randomly select 100 data points to display rng(2000); %random number generator seed
rng(2000); %random number generator seed
rand_indices = randperm(n);
sel = X(rand_indices(1:100), :);
Xdata = [ones(n, 1) X];

numberOfFolds = 5;
rng(2000); %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
method = 'KNN'

lambda = 0.1;
for i = 1:numberOfFolds
  TestIndex = find(CVindex == i);
  TrainIndex = find(CVindex ~= i);
  TrainDataCV = Xdata(TrainIndex,:);
  TrainDataGT =y(TrainIndex);
  TestDataCV = Xdata(TestIndex,:);
  TestDataGT = y(TestIndex);
  %
  %build the model using TrainDataCV and TrainDataGT %test the built model using TestDataCV
  %
  switch method
    case 'LogisticRegression'
      all_theta = zeros(num_labels, size(Xdata, 2));
      for c=1:num_labels
        initial_theta = zeros(size(Xdata, 2), 1);
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        [theta] = fmincg(@(t)(costFunctionLogisticRegression(t,TrainDataCV,(TrainDataGT == c),lambda)),initial_theta,options);
        all_theta(c,:) = theta;
      end
      all_pred = sigmoid(TestDataCV*all_theta');
      [maxVal,maxIndex] = max(all_pred,[],2);
      TestDataPred=maxIndex;
    case 'KNN'
      [id] = knnsearch(TrainDataCV,TestDataCV,'K',3);
      neighborM = [TrainDataGT(id(:,1)) TrainDataGT(id(:,2)) TrainDataGT(id(:,3))];
      freqM = mode(neighborM');
      TestDataPred = freqM';
    otherwise
      error('Unknown classification method')
  end
  predictionLabels(TestIndex,:) = double(TestDataPred);
end

confusionMatrix = confusionmat(y,predictionLabels);
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));
fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method,lambda,accuracy*100));
fprintf('Confusion Matrix:\n');
[r c] = size(confusionMatrix);
for i=1:r
  for j=1:r
    fprintf('%6d ',confusionMatrix(i,j));
  end
  fprintf('\n');
end
