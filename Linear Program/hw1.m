% CS 446 Homework 1
% Chongxin Luo

%data = readFeatures('hw1conjunctions.txt',10);
%data
%[w,theta,delta] = findLinearDiscriminant(data);
%w
%theta
%delta
data = readFeatures('hw1sample2d.txt', 2);
[w,theta,delta] = findLinearDiscriminant(data);
w
theta
delta
plot2dSeparator(w,theta);
plot2dData(data);