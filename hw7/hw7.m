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
% question 8
%==========================================================================
%Download hwk7files_forMycourses.zip and place them into an appropriate
%hwk7 directory.  
%Update the two paths below for your machine
 
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


fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(D, size(Theta1,1));
initial_Theta2 = randInitializeWeights(size(Theta1,1), C);
 % Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
 
fprintf('\nTraining Neural Network... \n')
 
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 5);
 
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
