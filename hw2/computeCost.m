function cost = computeCost(Xdata, Ydata, theta)
  % computes the cost
  cost = sum((Ydata-Xdata*theta).^2)/(2*length(Ydata));
end