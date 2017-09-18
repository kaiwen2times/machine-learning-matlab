function theta = gradientDescentMultiReg(Xdata, y, theta, alpha, num_iters, lam)
    n = length(y); % number of training examples
    temp = zeros(1,size(theta,1));

    for iter = 1:num_iters
        for i = 1:size(theta,1);
            temp(i) = theta(i)-(alpha/n)*sum((Xdata*theta-y).*Xdata(:,i))+(alpha/n)*abs(theta(i)*lam);
        end
        theta = temp';
    end
end
