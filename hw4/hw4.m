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
  hypo = ((Xdata*theta) >= threshold);
  result = (hypo == y);
  acc = sum(result)/length(y);
  str = strcat('\lambda=',num2str(lambda(index)),', Accuracy=',num2str(acc));
  title(str,'fontsize',10)
end
print('cmpe677_hwk4_2_circle','-dpng')
