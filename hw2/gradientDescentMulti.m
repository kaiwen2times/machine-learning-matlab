function [theta, J_history] = gradientDescentMulti(Xdata, y, theta, alpha, num_iters)
    n = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    temp = zeros(1,size(theta,1));

    for iter = 1:num_iters
        for i = 1:size(theta,1);
            temp(i) = theta(i) - (alpha/n)*sum((Xdata*theta-y).*Xdata(:,i));
        end
        theta = temp';
        % Save the cost J in every iteration    
        J_history(iter) = computeCost(Xdata, y, theta);
    end
end
