clear
redWineData=load('winequality-red.csv');


%Two sample solutions below, neither is good enough for points:
% scale each feature by finding the max in each column then divide
% each data point by the max
scaled = ones(1599,12);
for col=1:size(redWineData,2)
  val = max(redWineData(:,col));
  scaled(:,col) = redWineData(:,col)/val;
end
csvwrite('scaled.csv',scaled)
y = scaled(:,12);
M=[ones(length(redWineData),1) scaled(:,1:11) scaled(:,1:11).^2];  %avgSqErr=0.3996
% reminder: number of columns of M <=30

w = ((M'*M)\M')*y;
avgSqErr=sum((y-M*w).^2)./length(redWineData)
