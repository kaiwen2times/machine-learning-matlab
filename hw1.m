% question 1
[1 2 3]'; 
[1; 2; 3];

% question 2
A = [1 2 3; 2 3 4; 3 4 5; 4 5 6];
B = A(:,2:3);

% question 3
a = [1 2; 3 4];
b = [2 2; 3 3; 4 4];
c = eye(3);
d =[1 2 3];
%a*b
%a-b

% question 4
b = [2 2; 3 3; 4 4];
d =[1 2 3];
f = zeros(2,1);

%b - repmat(f,1,3);
%b + [f ; f; f];
%b + repmat(f',1,3);

% question 5
ones(4,1)*5;
ones(4,1).*5;
5*ones(1,4)';

% question 6
A = [1 2 3 4 5; 2 3 4 5 6; 3 4 5 6 7; 4 5 6 7 8; 5 6 7 8 9];
B=A(2:4,1:2:5);
B=A(2:4,[1 3 5]);
B=[A(2:4,1) A(2:4,3) A(2:4,5)];

% question 7
x=0:.05:(3/2)*pi;
figure
grid on
plot(sin(x),'r','LineWidth',3)
hold on
plot(sinc(x),'g','LineWidth',3)
hold on
plot(sin(x.^2),'LineWidth',3)
% labels
xlabel('time', 'FontSize',10)
ylabel('response', 'FontSize',10)
title('CMPE 677, Hwk 1, Problem 7, \lambda=0','FontSize',12)
legend('sine(x)','sinc(x)','sinc(x^{2})')
print –dpng C:\Users\kxz6582\Downloads\machine-intelligence\myFirstPlot.png

% question 8
A = [1 0 -4 8 3; 4 -2 3 3 1];
b = zeros(1,5);
for index = 1:size(A,2)
  if A(1,index) > A(2,index)
    b(index) = A(1,index);
  else
   b(index) = A(2,index);
  end
end
b;

% question 9
[eye(2) zeros(2,1)]*0
eye(2,3)*0
[0 0 0; 0 0 0]

% question 10
hold off
mu=[0 3];
sigma=[5 -2 ;-2 2];
x1 = -10:0.1:10; x2 = x1;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)], mu,sigma);
F = reshape(F,length(x2),length(x1));
contour(x1,x2,F);
grid on
axis square
title('CMPE 677, Hwk 1, Problem 10','fontsize',12);
print -dpng C:\Users\rwpeec\Desktop\rwpeec\rit\CMPE-
677_MachineIntelligence\hwk\hwk1\cmpe677_hwk1_10.png
