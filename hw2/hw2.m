% question 7
clear;
close all;
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
M = [ones(length(data),1) X];
W = ((M'*M)\M')*y
hy = X.*W(2)+W(1);
plot(X, y, 'rx', 'MarkerSize', 10,'LineWidth',3); % Plot the data
hold on
plot(X,hy,'g--', 'MarkerSize', 10,'LineWidth',3);
ylabel('Profit in $10,000s'); % Set the y axis label
xlabel('Population of City in 10,000s'); % Set the x axis label
grid on

% question 8
clear;
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
Xdata = [ones(length(X),1) X];
theta = zeros(2, 1); % initialize fitting parameters to zero
computeCost(Xdata,y,theta);