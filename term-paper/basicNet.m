% specifying input and output
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
y = [0,0,1,1]';

rng(1); % seed the random number generator
weights = rand(3,1); % there are only two hidden layers initialize one set of weights
err_sum = [];

for iter = 10:10:1000
  for i = 1:iter
    layer1 = X;
    layer2 = layer1 * weights;
    act = sigmoid(layer2);
    error = y - act;
    % (GT - prediction) * x * a * (1-a)
    delta = layer1' * (error .* (act .* (1-act)));
    weights = weights + delta;
  end
  err_sum = [err_sum, sum(error)];
end


plot(10:10:1000, err_sum)
xlabel('Iteration', 'FontSize',10)
ylabel('Total Error', 'FontSize',10)
title('Effect of Back Propagation','FontSize',12)
print('demo_nnet_1','-dpng')