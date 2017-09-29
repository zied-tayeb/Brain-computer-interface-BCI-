function ret = a(x,t,in)
    
   if(abs(sum(x-getPrevX)) > 1.0e-04)
       setX(x);
       disp(x);
       wipeE();
   end
    ret = (exp(x(5)*e(x,t,in)) - 1)/(exp(x(5))-1);
end

function ret = e(x,t,in)

    if t(1)<=x(4)
        ret = 0;
    else
        try
            ret = x(1)*in(int64(t(1)-x(4))) - x(2)*getE(t(1)-1) - x(3)*getE(t(1) -2);
        catch
            ret = x(1)*in(int64(t(1)-x(4))) - x(2)*e(x,t-1, in) - x(3)*e(x,t-2,in);
        end
    end
    updateE(ret, t(1));

end

function updateE(val,pos)
    global e_calc
    e_calc(pos) = val;
end

function ret  = getE(pos)
    global e_calc
    ret = e_calc(pos);
end

function wipeE()
    global e_calc;
    e_calc = [];
end

function setX(val)
    global X;
    X = val;
end

function ret = getPrevX()
    global X;
    ret = X;
end
