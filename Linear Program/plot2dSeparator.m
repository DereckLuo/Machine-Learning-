% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
% The 2D line equation: w1x+w2y+theta = 0
x = linspace(0,1.5);
y = (-theta - w(1)*x)/w(2);
plot(x,y);
end
