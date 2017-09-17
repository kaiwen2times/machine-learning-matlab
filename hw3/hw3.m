%===========================================================================
% question 1
%===========================================================================
clear;
close all;
x = [3 -2 5 1 0];


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
% Line going up means that a specific coefficient has influence on the cost
% function, the more influence it exerts, the higher the line. Line going
% down means that a specific coefficient has less influence on the cost
% function, the less influence it exerts, the lower the line.


%===========================================================================
% question 5
%===========================================================================
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
scatter(x, y,60,'MarkerEdgeColor','b','MarkerFaceColor','r')
hold on
plot(graphX, graphY,'b--','MarkerSize',10,'LineWidth',3)
% labels
title(str,'fontsize',14)
xlabel('Day of Year','fontsize',12); ylabel('Bank Acct. Balance','fontsize',12);
grid on
print('cmpe677_hwk3_5_5th_order','-dpng')


%===========================================================================
% question 6
%===========================================================================
%percent of way in semester
x = [0 0.2072 0.3494 0.4965 0.6485 0.7833 0.9400]';
%bank balance ($K)
y = [2.150 1.541 0.790 0.909 0.901 0.593 0.198]' ;
% 5th order linear regression
lambda = 0.001;
model = [ones(length(x),1) x x.^2 x.^3 x.^4 x.^5];
modelReg = [zeros(length(x),1) x x.^2 x.^3 x.^4 x.^5];
theta = ((model'*model)\model')*y;
theta2 = regularNormalEquation(x,y,modelReg,lambda);
% plot 
figure
graphX = (0:0.001:1)';
M2 = [ones(length(graphX),1) graphX graphX.^2 graphX.^3 graphX.^4 graphX.^5];
graphY = M2*theta;
scatter(x, y,60,'MarkerEdgeColor','b','MarkerFaceColor','r')
hold on
plot(graphX, graphY,'b--','MarkerSize',10,'LineWidth',3)
% labels
title(str,'fontsize',14)
xlabel('Percent of Way in Semester','fontsize',12); ylabel('Bank Balance ($K)','fontsize',12);
