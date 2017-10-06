%==========================================================================
% question 1
%==========================================================================
clear
close all
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
axis([0 10 0 12])
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
axis([0 10 0 12])
title('Descision Tree Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on

% Given n points, n is the maximum number of tree nodes to do perfect classification




%==========================================================================
% question 2
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
axis([0 10 0 12])
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
axis([0 10 0 12])
title('Descision Tree Classifier','fontsize',14)
xlabel('X','fontsize',12)
ylabel('Y','fontsize',12)
grid on


%==========================================================================
% question 3
%==========================================================================