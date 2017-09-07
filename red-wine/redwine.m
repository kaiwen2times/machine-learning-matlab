clear
redWineData=load('winequality-red.csv');

y = redWineData(:,12);
%Two sample solutions below, neither is good enough for points:



%M=[redWineData(:,1:8).^6 redWineData(:,1:11).^0.5 redWineData(:,1:11).^0.0005];  %avgSqErr=0.3996
% reminder: number of columns of M <=30

w = ((M'*M)\M')*y;
avgSqErr=sum((y-M*w).^2)./length(redWineData)