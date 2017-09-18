function coef = regularNormalEquation(xData, y, lambda)

inverse = xData'*xData;
m = eye(length(inverse));
m(1,1) = 0;
coef = ((inverse+m.*lambda)\xData')*y;

end
