%===========================================================================
% question 1
%===========================================================================
clear;
close all;
x = [3 -2 5 1 0];
x1 = length(x)-1
x2 = 3+2+5+1+0
x3 = 9+4+25+1
x4 = max(x)


%===========================================================================
% question 2
%===========================================================================
% b


%===========================================================================
% question 3
%===========================================================================
% 4 coefficients contribute at L1=5


%===========================================================================
% question 4
%===========================================================================
% Line going up means that a specific coefficient has postive correlation
% with the cost function, the more influence it exerts, the higher the line. Line going
% down means that a specific coefficient has negative correlation with the cost
% function


%===========================================================================
% question 5
%===========================================================================
clear
%day of year
x = [4 62 120 180 242 297 365]';
%bank balance
y = [2720 1950 1000 1150 1140 750 250]';

M = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5];
theta = ((M'*M)\M')*y;
avgSqErr = sum((y-M*theta).^2)./length(y);
err = num2str(avgSqErr,'%.5f');
str = strcat('5th order fit, \lambda=0.00000, avgSqErr=', err);
graphX = (1:400)';
M2 = [ones(length(graphX),1) graphX graphX.^2 graphX.^3 graphX.^4 graphX.^5];
graphY = M2*theta;
% plot
figure
scatter(x, y, 60,'MarkerEdgeColor','b','MarkerFaceColor','r')
hold on
plot(graphX, graphY,'b--','MarkerSize',10,'LineWidth',3)
% labels
title(str,'fontsize',14)
xlabel('Day of Year','fontsize',12);
ylabel('Bank Acct. Balance','fontsize',12);
grid on
print('cmpe677_hwk3_5_5th_order','-dpng')


%===========================================================================
% question 6
%===========================================================================
clear
%percent of way in semester
x = [0 0.2072 0.3494 0.4965 0.6485 0.7833 0.9400]';
%bank balance ($K)
y = [2.150 1.541 0.790 0.909 0.901 0.593 0.198]' ;
% 5th order linear regression
lambda = 0.001;
model = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5];
theta = ((model'*model)\model')*y;
theta2 = regularNormalEquation(model,y,lambda);
avgSqErr = sum((y-model*theta2).^2)./length(y);
str = strcat('5th order regularized fit, \lambda=', num2str(lambda), ', avgSqErr=', num2str(avgSqErr,'%.5f'));
% plot 
figure
graphX = (0:0.001:1)';
M1 = [ones(length(graphX),1) graphX graphX.^2 graphX.^3 graphX.^4 graphX.^5];
graphY1 = M1*theta;
graphY2 = M1*theta2;
scatter(x, y, 60,'MarkerEdgeColor','b','MarkerFaceColor','r')
hold on
plot(graphX, graphY1,'b--','MarkerSize',10,'LineWidth',3)
plot(graphX, graphY2,'m--','MarkerSize',10,'LineWidth',3)
% labels
title(str,'fontsize',14)
xlabel('Percent of Way in Semester','fontsize',12);
ylabel('Bank Balance ($K)','fontsize',12);
legend('Data Point', 'Unregularized', 'Regularized')
grid on
print('cmpe677_hwk3_6_regularized','-dpng')



%===========================================================================
% question 7
%===========================================================================
clear
%percent of way in semester
x =[0 0.2072 0.3494 0.4965 0.6485 0.7833 0.9400]';
%bank balance ($K)
y =[2.150 1.541 0.790 0.909 0.901 0.593 0.198]';
lambda = [0 0.001 1];
D = [1 3 5];
D1 = [ones(length(x),1) x];
D3 = [ones(length(x),1) x x.^2 x.^3];
D5 = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5];
models = {D1, D3, D5};
count = 0;
index = 0;
figure
for lam = lambda
  count = 0;
  for d = D
    index = index + 1;
    count = count + 1;
    str = strcat('D=', num2str(d), ', \lambda=', num2str(lam));
    subplot(3,3,index)
    theta = regularNormalEquation(models{count},y,lam);
    theta2 = ((models{count}'*models{count})\models{count}')*y;
    scatter(x, y, 20,'MarkerEdgeColor','b','MarkerFaceColor','r')
    hold on
    plot(x,models{count}*theta,'m--')
    plot(x,models{count}*theta2,'b--')
    title(str,'fontsize',8)
  end
end
print('cmpe677_hwk3_7_3x3','-dpng')
    


%===========================================================================
% question 8
%===========================================================================
clear
close all
figure
rng(2000);  %random number generator seed
mu = [0 0 ];
sigma = [4 1.5 ; 1.5 2];
r = mvnrnd(mu,sigma,50); %create two features, 50 samples of each
y = r(:,1);
x = (pi*(1:50)/20)';  %scale x for sin
y = 10*sin(x).*(4+y); % add some curvature
y = y + x*4;  % gradually rise over time
hold off;
plot(x,y,'x'); 

xtrain = x(1:2:end); ytrain = y(1:2:end);
xtest = x(2:2:end); ytest = y(2:2:end);
figure(1);
plot(xtrain, ytrain, 'rs', 'MarkerSize', 10,'LineWidth',3,'markerfacecolor','c','markeredgecolor','b'); % Plot the data
hold on
plot(xtest, ytest, 'ro', 'MarkerSize', 10,'LineWidth',3,'markerfacecolor','m','markeredgecolor','r'); % Plot the data
grid on;

model = [ones(length(x),1) x x.^2 x.^3];
[B, Stats] = lasso(model,y,'CV',2);
lassoPlot(B, Stats, 'PLotType','CV')
print('cmpe677_hwk3_8_lasso','-dpng')

% you could use less regularization because looking at the graph,
% the MSE decreases as lambda decreases.

one_se = B(:,Stats.Index1SE)
me = B(:,Stats.IndexMinMSE)
y_one_se = model*one_se;
y_me = model*me;
figure(1)
plot(x,y_one_se,'b--','MarkerSize',10,'LineWidth',3)
plot(x,y_me,'m--','MarkerSize',10,'LineWidth',3)
title('One Standard Error vs. Min Error','fontsize',14)
legend('train','test','One Standard Error Fit','Min Error Fit')
print('cmpe677_hwk3_8_ose_me','-dpng')


avgSqErrOSE = sum((y-y_one_se).^2)./length(y)
avgSqErrME = sum((y-y_me).^2)./length(y)




%===========================================================================
% question 9
%===========================================================================
clear
% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
 
% Scale features and set them to zero mean with std=1
[Xnorm mu sigma] = featureNormalize(X);  % reuse this function from hwk2
 
% Add intercept term to X
Xdata = [ones(length(X),1) Xnorm];
 
% Init Theta and lambda
theta = ((Xdata'*Xdata)\Xdata')*y;  %well..this is the optimal solution
lambda=1;
 
%Run Compute Cost 
disp('cost = ')
disp(computeCostReg(Xdata,y,theta, lambda))




%===========================================================================
% question 10
%===========================================================================
clear
data = load('ex1data1.txt'); % Dataset from Andrew Ng, Machine Learning MOOC
X = data(:, 1);
y = data(:, 2);
M = [ones(length(X),1) X];
theta_init = zeros(2, 1); % initialize fitting parameters to zero
% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
lambda = 0;
% run gradient descent
theta_unreg = gradientDescentMultiReg(M, y, theta_init, alpha, iterations, lambda);
lin_reg = ((M'*M)\M')*y; % optimal solution
lambda = 1;
theta_reg = gradientDescentMultiReg(M, y, theta_init, alpha, iterations, lambda);
fprintf('Linear Regression: [%f,%f]\n',lin_reg);
fprintf('Gradient Descent: [%f,%f]\n',theta_unreg);
fprintf('Regularized Gradient Descent: [%f,%f]\n',theta_reg)

lambda = 100;
theta_reg = gradientDescentMultiReg(M, y, theta_init, alpha, iterations, lambda);
fprintf('Regularized Gradient Descent with lambda=100: [%f,%f]\n',theta_reg)