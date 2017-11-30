% specifying input and output
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
y = [0,1,1,0]';

rng(1); % seed the random number generator
weights = rand(3,1); % there are only two hidden layers initialize one set of weights

for i = 1:20
  layer1 = X;
  layer2 = layer1 * weights;
  activation = sigmoid(layer2);
  error = y - activation;
  err_sum(i) = sum(error);
  % (GT - prediction) * x * a * (1-a)
  delta = layer1' * error * (activation' * (1-activation));
  weights = weights + delta;
end

plot(1:20,err_sum)