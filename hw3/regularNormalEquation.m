function coef = regularNormalEquation(xData, y, model, lambda)

coef = ((xData'*xData+model.*lambda)\xData')*y

end
