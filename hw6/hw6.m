clear;
clc;
close all;
%==========================================================================
% question 1
%==========================================================================
% given P(light) = 0.25, P(camera) = 0.2, and P(light,camera) = 0.05
% P(camera|light) = P(camera,light) / P(light) = 0.05 / 0.25 = 0.2



%==========================================================================
% question 2
%==========================================================================
% P(A,B,C,D,E,G)
% = P(G|A,B,C,D,E) * P(A,B,C,D,E)
% = P(G|E) * P(E|A,B,C,D) * P(A,B,C,D)
% = 0.5 * P(E|C) * P(D|A,B,C) * P(A.B.C)
% = 0.5 * 0.1 * P(D|B,C) * P(A.B.C)
% = 0.5 * 0.1 * 0.5 * P(C|A.B) * P(A.B)
% = 0.5 * 0.1 * 0.5 * P(C|A) * P(B|A) * P(A)
% = 0.5 * 0.1 * 0.5 * 0.8 * 0.1 * 0.5
% = 0.001


% P(B,C)
% = P(B,C,A) + P(B,C,A')
% = P(B|C,A) * P(C,A) + P(B|C,A') * P(C,A')
% = P(B|A) * P(C,A) + P(B|A') * P(C,A')
% = 0.1 * P(C|A) * P(A) + 0.5 * P(C|A') * P(A')
% = 0.1 * 0.8 * 0.5 + 0.5 * 0.1 * 0.5
% = 0.065



%==========================================================================
% question 3
%==========================================================================
% P(A,B',C,D,E',F,I) follows a similar pattern to previous question
% = P(I|F) * P(F|D) * P(E'|C) * P(D|B',C) * P(C|A) * P(B?|A) * P(A)



%==========================================================================
% question 4
%==========================================================================
% Pitch at 14K, w1 = kitten, W2 = puppy
% P(W1|14K) = P(14K|W1) * P(W1) / P(14K)
% P(W2|14K) = P(14K|W2) * P(W2) / P(14K)
% Compare P(W1|14K) with P(W2|14K), the denominator can be cancelled out
% It reduces to P(14K|W1) * P(W1) ? P(14K|W2) * P(W2)
% Looking at the graph
% P(14K|W1) = 0.1, P(W1) = 3 / 30 = 0.1
% P(14K|W2) = 0.06, P(W2) = 27 / 30 = 0.9
% P(14K|W1) * P(W1) = 0.1 * 0.1 = 0.01
% P(14K|W2) * P(W2) = 0.06 * 0.9 = 0.054
% It's more likely to be a puppy because the there is way more puppies that
% kittens



%==========================================================================
% question 5
%==========================================================================
% Row 1
% P(A|B',E',J',M')
% = P(A,B',E',J',M') / P(B',E',J',M')
% = P(J'|A) * P(M'|A) * P(A|B',E') * P(B') * P(E') / P(B',E',J',M')
% = 0.1 * 0.2 * 0.2 * 0.9 * 0.8 / (P(B',E',J',M',A) + P(B',E',J',M',A'))
% = 0.0029 / (0.0029 + P(J'|A') * P(M'|A') * P(A'|B',E') * P(B') * P(E'))
% = 0.0029 / (0.0029 + 0.8 * 0.9 * 0.8 * 0.9 * 0.8)
% = 0.0029 / (0.0029 + 0.4147)
% = 0.0069

% Row 2
% P(A|B',E',J,M')
% = P(A,B',E',J,M') / P(B',E',J,M')
% = P(J|A) * P(M'|A) * P(A|B',E') * P(B') * P(E') / P(B',E',J,M')
% = 0.9 * 0.2 * 0.2 * 0.9 * 0.8 / (P(B',E',J,M',A) + P(B',E',J,M',A'))
% = 0.0259 / (0.0259 + P(J|A') * P(M'|A') * P(A'|B',E') * P(B') * P(E'))
% = 0.0259 / (0.0259 + 0.2 * 0.9 * 0.8 * 0.9 * 0.8)
% = 0.0259 / (0.0259 + 0.1037)
% = 0.1998

% Row 3
% P(A|B,E',J,M')
% = P(A,B,E',J,M') / P(B,E',J,M')
% = P(J|A) * P(M'|A) * P(A|B,E') * P(B) * P(E') / P(B,E',J,M')
% = 0.9 * 0.2 * 0.6 * 0.1 * 0.8 / (P(B,E',J,M',A) + P(B,E',J,M',A'))
% = 0.0086 / (0.0086 + P(J|A') * P(M'|A') * P(A'|B,E') * P(B) * P(E'))
% = 0.0086 / (0.0086 + 0.2 * 0.9 * 0.4 * 0.1 * 0.8)
% = 0.0086 / (0.0086 + 0.0058)
% = 0.5972

% Row 4
% P(A|B',E',J',M)
% = P(A,B',E',J',M) / P(B',E',J',M)
% = P(J'|A) * P(M|A) * P(A|B',E') * P(B') * P(E') / P(B',E',J',M)
% = 0.1 * 0.8 * 0.2 * 0.9 * 0.8 / (P(B',E',J',M,A) + P(B',E',J',M,A'))
% = 0.0115 / (0.0115 + P(J'|A') * P(M|A') * P(A'|B',E') * P(B') * P(E'))
% = 0.0115 / (0.0115 + 0.8 * 0.1 * 0.8 * 0.9 * 0.8)
% = 0.0115 / (0.0115 + 0.0461)
% = 0.1997

% Row 5
% P(A|B',E,J,M')
% = P(A,B',E,J,M') / P(B',E,J,M')
% = P(J|A) * P(M'|A) * P(A|B',E) * P(B') * P(E) / P(B',E,J,M')
% = 0.9 * 0.2 * 0.3 * 0.9 * 0.2 / (P(B',E,J,M',A) + P(B',E,J,M',A'))
% = 0.0097 / (0.0097 + P(J|A') * P(M'|A') * P(A'|B',E) * P(B') * P(E))
% = 0.0097 / (0.0097 + 0.2 * 0.9 * 0.7 * 0.9 * 0.2)
% = 0.0097 / (0.0097 + 0.0227)
% = 0.2994

% Row 6
% P(A|B',E',J',M)
% = P(A,B',E',J',M) / P(B',E',J',M)
% = 0.0115 / (0.0115 + 0.0461)
% = 0.1997

% Row 7
% P(A|B,E,J,M)
% = P(A,B,E,J,M) / P(B,E,J,M)
% = P(J|A) * P(M|A) * P(A|B,E) * P(B) * P(E) / P(B,E,J,M)
% = 0.9 * 0.8 * 0.8 * 0.1 * 0.2 / (P(B,E,J,M,A) + P(B,E,J,M,A'))
% = 0.0115 / (0.0115 + P(J|A') * P(M|A') * P(A'|B,E) * P(B) * P(E))
% = 0.0115 / (0.0115 + 0.2 * 0.1 * 0.1 * 0.1 * 0.2)
% = 0.0115 / (0.0115 + 0.00004)
% = 0.9965

% Row 8
% P(A|B',E',J',M')
% = P(A,B',E',J',M') / P(B',E',J',M')
% = 0.0029 / (0.0029 + 0.4147)
% = 0.0069

% Row 9
% P(A|B',E',J,M')
% = P(A,B',E',J,M') / P(B',E',J,M')
% = 0.0259 / (0.0259 + 0.1037)
% = 0.1998

% Row 10
% P(A|B',E',J',M)
% = P(A,B',E',J',M) / P(B',E',J',M)
% = 0.0115 / (0.0115 + 0.0461)
% = 0.1997



%==========================================================================
% question 6
%==========================================================================
% P(A|BE) = 0.9 / 1 = 0.9

% P(A|B'E) = 0.2 / 1 = 0.2

% P(A|BE') = 0.8 / 1 = 0.8

% P(A|B'E') = (0.01 + 0.1 + 0.1 + + 0.1 + 0.01 + 0.1 + 0.1) / 7 = 0.0743

% P(J|A) = 2 / 2 = 1 

% P(J|A') = 3 / (10 - 2) = 3 / 8 = 0.3750

% P(M|A) = 1 / 2 = 0.5 

% P(M|A') = 3 / 8 = 0.3750



%==========================================================================
% question 7
%==========================================================================
% SVM decision boundary is only affected by support vectors, while logistic
% regression takes all data points into account.



%==========================================================================
% question 9
%==========================================================================
addpath('C:\Users\kxz6582\Downloads\machine-intelligence\hw6\libsvm-3.18\windows');
load('ex6data1.mat'); %load Andrew Ng data
% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
figure
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;
plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
% Cost value
C = 25;
% Call svmtrain from LIBSVM
% -t 0 says to do a linear kernel
% -c <value> sets cost to <value>, higher c means SVM will weigh errors more
eval(['model = svmtrain(y,X,''-t 0 -c ' num2str(C) ''');']);
% SVM solves wx+b
% alpha values are stored in model.sv_coeff
% Note: alpha values are really y(i) * alpha(i)- so no need to include y when solving for w
% support vectors are stored in model.SVs
% solve for w here:
% w = <insert code here>
w = (model.sv_coef' * full(model.SVs)); %full converts from sparse to full matrix representation
% b is stored in -model.rho
b = -model.rho;
%plot boundary ontop of data
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
hold on;
plot(xp, yp, 'b:','linewidth',3);
%Predictions are sign(<x,w> + b), note: can do all predictions in one line
%predictionsTrain = <insert code here>
predictionsTrain = sign(X * w' + b);
predictionsTrain(predictionsTrain==-1) = 0; %change -1 to 0 to match GT
%compute training error
%predictionsTrainError = <insert code here>
predictionsTrainError = sum(predictionsTrain~= y)/length(y);
fprintf('Error on train set = %0.2f%%\n',predictionsTrainError*100);
%Now we will see how this does on a test set
Xtest = [ 1 3; 2 3; 3 3; 4 3; 1 4; 2 4; 3 4; 4 4];
ytest = [0 0 0 1 0 1 1 1]';
%predictionsTest = <insert code here>
predictionsTest = sign(Xtest * w' + b);
predictionsTest(predictionsTest==-1) = 0; %change -1 to 0 to match GT
%predictionsTest = <insert code here>
predictionsTestError = sum(predictionsTest ~= ytest)/length(ytest);
fprintf('Error on test set = %0.2f%%\n',predictionsTestError*100);
print('cmpe677_hwk6_9_svm','-dpng')


% What is D?
% 2

% What is n?
% 51

% How many classes are there?
% 2

% How many support vectors are there?
% 12

% What is the Error on the train set?
% 1.96%

% What is the Error on the test set?
% 12.50%

% What is the smallest integer value of the Cost value C, such that the train error is 0%?
% 25



%==========================================================================
% question 10
%==========================================================================
clear;
addpath('C:\Users\kxz6582\Downloads\machine-intelligence\hw6\libsvm-3.18\windows');
load('ex6data2.mat'); %load Andrew Ng data
% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
figure
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;
plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
% Cost value
C = 1;
% Call svmtrain from LIBSVM
% -t 0 says to do a linear kernel
% -c <value> sets cost to <value>, higher c means SVM will weigh errors more
eval(['model = svmtrain(y,X,''-t 0 -c ' num2str(C) ''');']);
% SVM solves wx+b
% alpha values are stored in model.sv_coeff
% Note: alpha values are really y(i) * alpha(i)- so no need to include y when solving for w
% support vectors are stored in model.SVs
% solve for w here:
% w = <insert code here>
w = (model.sv_coef' * full(model.SVs)); %full converts from sparse to full matrix representation
% b is stored in -model.rho
b = -model.rho;
%plot boundary ontop of data
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
hold on;
plot(xp, yp, 'b:','linewidth',3);
%Predictions are sign(<x,w> + b), note: can do all predictions in one line
%predictionsTrain = <insert code here>
predictionsTrain = sign(X * w' + b);
predictionsTrain(predictionsTrain==-1) = 0; %change -1 to 0 to match GT
%compute training error
%predictionsTrainError = <insert code here>
predictionsTrainError = sum(predictionsTrain~= y)/length(y);
fprintf('Error on train set = %0.2f%%\n',predictionsTrainError*100);
%Now we will see how this does on a test set
Xtest = [ 1 3; 2 3; 3 3; 4 3; 1 4; 2 4; 3 4; 4 4];
ytest = [0 0 0 1 0 1 1 1]';
%predictionsTest = <insert code here>
predictionsTest = sign(Xtest * w' + b);
predictionsTest(predictionsTest==-1) = 0; %change -1 to 0 to match GT
%predictionsTest = <insert code here>
predictionsTestError = sum(predictionsTest ~= ytest)/length(ytest);
fprintf('Error on test set = %0.2f%%\n',predictionsTestError*100);
print('cmpe677_hwk6_10_svm','-dpng')



%==========================================================================
% question 11
%==========================================================================
clear;
addpath('C:\Users\kxz6582\Downloads\machine-intelligence\hw6\libsvm-3.18\windows');
load('ex6data2.mat'); %load Andrew Ng data
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
figure
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;

plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
% Cost value
C = 1;
%Now try a radial basis function
% Call svmtrain from LIBSVM
% -t 2 says to do a radial basis kernel
% -c <value> sets cost to <value>, higher c means SVM will weigh errors more
% -g <value> sets gamma to <value>, higher g means smoother fit
% Try a linear kernel first
% gamma value
for g=0:10:1000
    eval(['model = svmtrain(y,X,''-t 2 -c ' num2str(C) ' -g ' num2str(g) ''' );']);
    %Predictions
    predictionsTrain = svmpredict( y, X, model, '-q');
    predictionsTrain(predictionsTrain==-1) = 0; %change -1 to 0 to match GT

    %compute training error
    predictionsTrainError = sum(predictionsTrain~= y)/length(y);
    fprintf('Error on train set = %0.2f%%\n',predictionsTrainError*100);
    if predictionsTrainError == 0
    break
end
end
fprintf('The min value of g for 0%% training error is: %d\n',g);
visualizeBoundary2D(X, y, model);
str = sprintf('The min value of g for 0%% training error is: %d\n',g);
title(str,'fontsize',14);
print -dpng hwk6_q7.png
