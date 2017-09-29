function ret = e(x,k,arr)
%     ret = x(1)*arr(t-x(4)) - x(2)*func(x,t-1) - x(3)*e(x,t-2);
    ret = x*arr(k) - x*e(k-1,arr) - x*e(k-2,arr);

end

