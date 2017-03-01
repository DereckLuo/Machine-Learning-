% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
% m : # of data points, n : # of variables in a data point
[m, np1] = size(data);
n = np1-1;


% write your code here: find c, A and b
%        1
% b - < ... >
%        0
b = ones(m,1);  % b is (m+1 x 1)
b = [b(1:m,:); [0]; b(m+1:end,:)];

% c - < 0 0 ... 0 1 >
c = zeros(1,n); % c is (1 x n+2)
c = [c [0 1]];

%       y1x1 y1x2 ... y1xn y1 1 
% a - <           ...           >
%       ymx1 ymx2 ... ymxn ym 1
%         0   0   ...   0   0 1
%
% Construct matrix A: A is (m+1 x n+2)
A = zeros(m,n);
last_column = ones(m, 1);
sl_column = zeros(m,1);
last_row = [zeros(1,n+1) [1]];
for i = 1:m
    yi = data(i + m*n);
    sl_column(i) = yi;
    for j = 1:n
        idx = i + m*(j-1);
        A(idx) = data(idx)*yi;
    end
end
A = [A sl_column last_column];
A = [A(1:m,:); last_row; A(m+1:end,:)];

%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);


end
