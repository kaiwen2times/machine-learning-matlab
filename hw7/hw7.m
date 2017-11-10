clear;
clc;
close all;
%==========================================================================
% question 1
%==========================================================================
% a)What is the dimension of the input features?
% ans: 3

% b)Would this work better as a happy-neutral-sad or happy-sad classifier?
% ans: happy-sad classifier

% c)What is the dimension of the y(2)?
% ans: 2

% d)x0(1) and a0(2) are called?
% ans: bias terms

% e)If the softmax function is used for the output layer, how many nodes use activation functions?
% ans: 3

% f)If you had training samples, {xi,yi}, i=1...10,000, would you pre-train with an unsupervised technique?
% ans: no



%==========================================================================
% question 2
%==========================================================================
% AND function
% x1 XOR x2 = (x1 OR x2)  AND  (NOT(x1 AND x2))
% a1 = -20, a2 = -20, w = 30




%==========================================================================
% question 3
%==========================================================================
% b, d, e, g, h, j



%==========================================================================
% question 4
%==========================================================================
% ReLu



%==========================================================================
% question 5
%==========================================================================
% in proportion to input values, because all of the information we have is
% the input values.



%==========================================================================
% question 6
%==========================================================================
% (yi - ai) is the difference between the prediction and the ground truth.
% a * (1 - a) is for the activation function
% xj is one of the input or a neuron.



%==========================================================================
% question 7
%==========================================================================
% a, d, f, i



%==========================================================================
%% question 8a, 8b
%==========================================================================
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory.  
%Update the two paths below for your machine
close all; clear all;
% We will use mnist hand written digits, '0' through '9'
load('ex4data1.mat');  %5000 Mnist digits from Andrew Ng class
n = size(X, 1);  %number of samples = 5000, 500 from each class
D = size(X,2);  %number of dimensions/sample.  20x20=400
C = length(unique(y));  %number of classes, class values are 1...10, where 10 is a digit '0'
 
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));  %This function is from Andrew Ng's online class
 
%Convert X and y data into Matlab nnet format:
inputs = X';
%one-hot encoding ground truth values 
targets = zeros(C,n);
for ii=1:n
  targets(y(ii),ii) =  1;
end
%If given one-hot encoding, can convert back to vector of ground truth
%class values:
% target1Dvector=zeros(n,1);
% for ii=1:n
%         target1Dvector(ii) = find(targets(:,ii) == 1);
% end
% max(target1Dvector-y) %this will be zero
fprintf('\nLoading Saved Neural Network Parameters ...\n')
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');  %Pre-learned weights from Andrew Ng class
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

lambda = 0;
J = nnCostFunction(nn_params, D, size(Theta1,1), C, X, y, lambda);
fprintf(['Cost at parameters (no regularization): %f \n'], J);

lambda = 1;
J = nnCostFunction(nn_params, D, size(Theta1,1), C, X, y, lambda);
fprintf(['Cost at parameters (with regularization): %f \n'], J);


%==========================================================================
% question 8c
%==========================================================================
% params
input_layer_size = D;
hidden_layer_size = size(Theta1,1);
num_labels = C;

% back propagation
fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(D, size(Theta1,1));
initial_Theta2 = randInitializeWeights(size(Theta1,1), C);
 % Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
 
fprintf('\nTraining Neural Network... \n')
 
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);
 
%  You can try different values of lambda, but keep lambda=1 for this exercise
lambda = 1;
 
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                    input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
 
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 
% Visual weights- can uncomment out next two lines to see weights            
% fprintf('\nVisualizing Neural Network... \n')
% displayData(Theta1(:, 2:end));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Training Set Accuracy: 94.960000


%==========================================================================
%% question 9a
%==========================================================================
close all; clear all;
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory.  Update the two paths below for your machine
 
% We will use mnist hand written digits, '0' through '9'
load('ex4data1.mat');  %5000 Mnist digits from Andrew Ng class
n = size(X, 1);  %number of samples = 5000, 500 from each class
D = size(X,2);  %number of dimensions/sample.  20x20=400
C = length(unique(y));  %number of classes, class values are 1...10, where 10 is a digit '0'
 
%Convert X and y data into Matlab nnet format:
inputs = X';
%one-hot encoding ground truth values 
targets = zeros(C,n);
for ii=1:n
    targets(y(ii),ii) =  1;
end

% Create a Pattern Recognition Network, with one hidden layer containing 25
% nodes
hiddenLayerSize = 25;
setdemorandstream(2014784333);   %seed for random number generator
net = patternnet(hiddenLayerSize);
 
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 0.7;  %note- splits are done in a random fashion
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
 
% Train the Network
[net,tr] = train(net,inputs,targets);  %return neural net and a training record
plotperform(tr); %shows train, validation, and test per epoch
 
% Test the returned network on the testing split
testX = inputs(:,tr.testInd); 
testT = targets(:,tr.testInd);
testY = net(testX);   %pass input test values into network
testIndices = vec2ind(testY);  %converts nnet float for each class into most likely class per sample
figure; plotconfusion(testT,testY)
[c,cm] = confusion(testT,testY);
 
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));  %Should be approx 91.6%
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);  %Should be approx 8.4%
%Should be approx 91.6%
 
% If your nnet had 2 class output, try this set:
%[inputs,targets] = cancer_dataset;  %breast cancer dataset built into Matlab
% you can use the receiver
% operating characteristic (ROC) plot to measure of how well 
% the neural network has fit data is the data.  This shows how the 
% false positive and true positive rates relate as the thresholding of 
% outputs is varied from 0 to 1.
% The farther left and up the line is, the fewer false positives need to 
% be accepted in order to get a high true positive rate. The best classifiers 
% will have a line going from the bottom left corner, to the top left corner, 
% to the top right corner, or close to that.
% Note: for this to work, testY needs to be a softMax, or similar
% continuous ouput
%plotroc(testT,testY)
 
% View the Network
view(net); %shows each layer with number of inputs and outputs to each



%==========================================================================
%% question 9b
%==========================================================================
close all; clear all;
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory.  Update the two paths below for your machine
addpath('C:\Users\kxz6582\Downloads\machine-intelligence\hw6\libsvm-3.18\windows')
 
% We will use mnist hand written digits, '0' through '9'
load('ex4data1.mat');  %5000 Mnist digits from Andrew Ng class
n = size(X, 1);  %number of samples = 5000, 500 from each class
D = size(X,2);  %number of dimensions/sample.  20x20=400
C = length(unique(y));  %number of classes, class values are 1...10, where 10 is a digit '0'
 
options.numberOfFolds = 5;
options.method = 'SVM';
[confusionMatrix_svm,accuracy_svm] =  classify677_hwk7(X,y,options);
 
options.method = 'nnet';
options.nnet_hiddenLayerSize = 25;
[confusionMatrix_nnet1,accuracy_nnet1] =  classify677_hwk7(X,y,options);
 
fprintf('Linear SVM accuracy is: %0.2f%%\n',accuracy_svm*100);
fprintf('Nnet accuracy with %d hidden layers, num nodes per layer = %d is: %0.2f%%\n',length(options.nnet_hiddenLayerSize),options.nnet_hiddenLayerSize,accuracy_nnet1*100);
 
options.method = 'nnet';
options.nnet_hiddenLayerSize = [25 10];
[confusionMatrix_nnet2,accuracy_nnet2] =  classify677_hwk7(X,y,options);
fprintf('Nnet accuracy with %d hidden layers, num nodes per layer = [%d %d] is: %0.2f%%\n',length(options.nnet_hiddenLayerSize),options.nnet_hiddenLayerSize,accuracy_nnet2*100);


% Total nSV = 1344
% SVM: Accuracy =  90.68% 
% Confusion Matrix:
%    488      3      0      4      2      0      1      0      2      0 
%      6    453      3     11      4      3      6      7      0      7 
%      6     12    437      0     25      0      5      8      6      1 
%      3      4      2    462      1      3      1      2     18      4 
%      3      7     29      5    427      6      0     12      6      5 
%      3      5      0      7      5    474      0      2      0      4 
%      6      6      4     19      1      0    441      0     23      0 
%     13     18     17      3     14      5      1    423      2      4 
%      2      4      5     18      3      1     21      3    440      3 
%      0      4      1      3      1      2      0      0      0    489 
% nnet: Accuracy =  90.98% 
% Confusion Matrix:
%    479      1      3      1      2      0      2     10      2      0 
%      5    433     13     10      8      7      8     12      1      3 
%      2     13    448      0     17      2      9      5      4      0 
%      3      3      1    460      1      9      2      3     17      1 
%      2      4     22      4    432      8      2     14      6      6 
%      3     13      0      3      8    464      0      4      0      5 
%      5      4      0     10      0      0    463      4     12      2 
%      5     12     10      4      9      7      3    442      6      2 
%      3      2      5     11      5      0     19      3    448      4 
%      0      0      4      2      3      2      1      6      2    480 
% Linear SVM accuracy is: 90.68%
% Nnet accuracy with 1 hidden layers, num nodes per layer = 25 is: 90.98%
% nnet: Accuracy =  89.44% 
% Confusion Matrix:
%    473      3      1      2      9      1      6      5      0      0 
%      5    429     16      8      3      7     11     16      1      4 
%      4     12    431      2     25      2      7      8      6      3 
%      4      3      1    452      3     10      2      4     21      0 
%      1      3     24      5    426     12      0     15      9      5 
%      1     10      0      7     11    462      2      3      1      3 
%      8      3      4      8      0      0    455      1     20      1 
%     12      9     16      6     20      6      3    419      2      7 
%      2      0      5     18      2      1     14      6    448      4 
%      0      3      3      1      4      6      3      2      1    477 
% Nnet accuracy with 2 hidden layers, num nodes per layer = [25 10] is: 89.44