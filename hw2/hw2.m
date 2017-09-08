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
% question 11
%===========================================================================
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));
% Fill out J_vals
for i = 1:length(theta0_vals)
  for j = 1:length(theta1_vals)
    t = [theta0_vals(i); theta1_vals(j)];
    J_vals(i,j) = computeCost(M, y, t);
  end
end
% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals,     J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
% print -dpng surfaceCost.png

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 1000
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
% print -dpng surfaceContour.png



%===========================================================================
% question 12
%===========================================================================
clear;
close all;
% Load Data
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