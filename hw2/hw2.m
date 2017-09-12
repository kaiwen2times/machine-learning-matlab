%===========================================================================
% question 7
%===========================================================================
clear;
close all;
figure
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
M = [ones(length(data),1) X];
W = ((M'*M)\M')*y;
hy = M*W; 
plot(X, y, 'rx', 'MarkerSize', 10,'LineWidth',3); % Plot the data
hold on
plot(X,hy,'g:', 'MarkerSize', 10,'LineWidth',3);
ylabel('Profit in $10,000s'); % Set the y axis label
xlabel('Population of City in 10,000s'); % Set the x axis label
grid on


%===========================================================================
% question 8
%===========================================================================
clear;
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
Xdata = [ones(length(X),1) X];
theta = zeros(2, 1); % initialize fitting parameters to zero
computeCost(Xdata,y,theta);


%===========================================================================
% question 9
%===========================================================================
figure
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
M = [ones(length(X),1) X];
theta_init = zeros(2, 1); % initialize fitting parameters to zero
% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
% run gradient descent
theta = gradientDescentLinear(M, y, theta_init, alpha, iterations);
plot(X, y, 'rx', 'MarkerSize', 10,'LineWidth',3); % Plot the data
hold on
W = ((M'*M)\M')*y;
gy = M*theta;
ly = M*W;
plot(X,gy, 'MarkerSize', 10,'LineWidth',3);
plot(X,ly, 'g:', 'MarkerSize', 10,'LineWidth',3);
ylabel('Profit in $10,000s'); % Set the y axis label
xlabel('Population of City in 10,000s'); % Set the x axis label
grid on


%===========================================================================
% question 10
%===========================================================================
data1 = [1, 3.5];
data2 = [1, 7];
ly1 = data1*W
gy1 = data1*theta
ly2 = data2*W
gy2 = data2*theta


%===========================================================================
% question 12
%===========================================================================

% Load Data
clear;
close all;
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version
% of X, where we substract the mean form each feature,
% then scale so that std dev = 1
[Xnorm mu sigma] = featureNormalize(X);
% Add intercept term to X
Xdata = [ones(length(X),1) Xnorm];
% Choose some alpha value
alpha = 0.01;
num_iters = 400;
% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(Xdata, y, theta, alpha, num_iters);
plot(1:400,J_history, 'MarkerSize', 10,'LineWidth',3);
plot(X,ly, 'MarkerSize', 10,'LineWidth',3);
ylabel('Cost'); % Set the y axis label
xlabel('Iteration'); % Set the x axis label
grid on