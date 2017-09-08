function cost = computeCost(Xdata, Ydata, theta)
  cost = sum((Ydata-Xdata*theta).^2)/(2*length(Ydata));
end