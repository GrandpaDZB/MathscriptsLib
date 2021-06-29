clc;clear
disp("======== Mode ========");
disp("1 Linear Regression");
disp("2 Polynomial Curve Fitting");
disp("3 DIY");
disp("=======================");

mode = input('Mode = ');

switch (mode)
   
    case 1
        y = input('(y1,...,yn)T y = ');
        X = input('(x1T,...,xnT) X = ');
        w = (X'*X)\X'*y;
        disp(w);
        
    case 2
        y = input('(y1,...,yn)T y = ');
        x = input('(x1,...,xn)T x = ');
        n = input('dim = ');
        X = ones(length(x),n+1);
        for i=1:n
            for j = 1:length(x)
                X(j,i) = x(j)^(n - i + 1);
            end
        end
        w = (X'*X)\X'*y;
        disp(w);
       
    case 3
        y = input('(y1,...,yn)T y = ');
        X = input('(x1T,...,xnT) X = ');
        w = (X'*X)\X'*y;
        disp(w); 
end