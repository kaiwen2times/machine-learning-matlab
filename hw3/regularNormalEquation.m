function coef = regularNormalEquation(xData, y, lambda)

inverse = xData'*xData;
m = [zeros(length(inverse),1) ones(length(inverse),size(inverse,2)-1)];
coef = ((inverse+m.*lambda)\xData')*y;

end
