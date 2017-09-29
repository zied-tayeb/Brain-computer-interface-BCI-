function Y = lpfilter(X,Fc,Fs,type,n)
%
%lpfilter design and apply low-pass iir filter coefficients for Butterworth
%or critically damped filters.
%
%by Cédric Morio - April 2009
%Oxylane Research - Department of Movement Sciences
%
%   Y = lpfilter(X, Fc, Fs) Filters the X signal with a 4th order 
%   Butterworth zero-lag filter (2 passes of a 2nd order). Returns the Y
%   filtered signal. Fc is the cut-off frequency and Fs the sampling
%   frequency of the original signal.
%
%   Y = lpfilter(X, Fc, Fs, TYPE) Filters the X signal with a 4th order 
%   Butterworth zero-lag filter (1 two-passes of a 2nd order) if TYPE is
%   'butter' and filters the X vector with a 20th order critically-damped
%   filter zero-lag too (5 two-passes of a second order) if TYPE is
%   'damped'. Returns the Y filtered signal. These calculations are in 
%   agreement with Robertson & Dowling recommendations [1].
%
%   Y = lpfilter(X, Fc, Fs, TYPE, N) Filters the X signal with a 2nd order
%   filter 'butter' or 'damped' with N passes. If the number of passes N is
%   impair for a simple filter and if it is pair for a zero-lag phase 
%   filter. Returns the Y filtered signal.
%
%   References:
%   [1] D.G.E. Robertson, J.J. Dowling. Design and responses of Butterworth and
%   critically damped digital filters. Journal of Electromyography and
%   Kinesiology, vol 13, 2003, pp566-573.
%   [2] D.A. Winter. Biomechanics and Motor Control of Human Movement. 2nd
%   edition, 1990, pp36-41.
%
%   See also BUTTER, FILTER, FILTFILT
if nargin < 3
    error('not enough arguments when calling lpfilter.');
elseif nargin > 5
    error('too much arguments when calling lpfilter.');
elseif nargin == 3
    type = 'butter'; % type of filter is not specified
    n = 2; % number of passes is not specified
elseif nargin == 4
    switch type
        case 'butter'
            n = 2;
        case 'damped'
            n = 10;
        otherwise
            error('in lpfilter the low-pass filter type must be ''butter'' or ''damped''.');
    end
end
if n < 1
    error('in lpfilter the number of passes n must be an positive integer.');
end
if Fc <= 0 || Fs <= 0
    error('in lpfilter the cutoff frequency Fc and the sampling rate Fs must be positive.');
end
switch type
    case 'butter'
        % design the butterworth filter coefficients
        [b,a] = butterworth(n,Fc,Fs);
    case 'damped'
        % design the critically damped filter coeefficients
        [b,a] = criticallydamped(n,Fc,Fs);
    otherwise
        error('in lpfilter the low-pass filter type must be ''butter'' or ''damped''.');
end
F = X;
while n > 1
    F = doublepass(b,a,F);
    n = n - 2;
end
if mod(n,2) && n > 0
    % cas N impair
    F = tofilt(b,a,F);
end
Y = F;
%--------------------------------------------------------------------------
function [b,a] = butterworth(n,Fcut,Fsr)
% This internal function returns num/den coefficient for the butterworth
% filter.
if n == 1
% if the number of passes of the Butterworth is 1, there is no correction
    Cbw = 1;
else
% the correction is :
    Cbw = 1/(nthroot((2^(1/n))-1,4));
end
Fbw = Fcut * Cbw;
Wc = tan((pi*Fbw)/Fsr);
K1 = sqrt(2)*Wc;
K2 = Wc^2;
b(1) = K2/(1+K1+K2); % a0 in Robertson & Dowling 2003
b(2) = 2*b(1);% a1 in Robertson & Dowling 2003
b(3) = b(1); % a2 in Robertson & Dowling 2003
a(1) = 1; % b0 in Robertson & Dowling 2003
a(2) = 2*b(1)*((1/K2)-1); % b1 in Robertson & Dowling 2003
a(3) = 1-(b(1)+b(2)+b(3)+a(2)); % b2 in Robertson & Dowling 2003
%--------------------------------------------------------------------------
function [b,a] = criticallydamped(n,Fcut,Fsr)
% This internal function returns num/den coefficient for the critically
% damped filter.
Ccrit = 1/(sqrt((2^(1/(2*n)))-1)); % correction of the cutoff frequency
Fcrit = Fcut * Ccrit;
Wc = tan((pi*Fcrit)/Fsr);
K1 = 2*Wc;
K2 = Wc^2;
b(1) = K2/(1+K1+K2); % a0 in Robertson & Dowling 2003
b(2) = 2*b(1);% a1 in Robertson & Dowling 2003
b(3) = b(1); % a2 in Robertson & Dowling 2003
a(1) = 1; % b0 in Robertson & Dowling 2003
a(2) = 2*b(1)*((1/K2)-1); % b1 in Robertson & Dowling 2003
a(3) = 1-(b(1)+b(2)+b(3)+a(2)); % b2 in Robertson & Dowling 2003
%--------------------------------------------------------------------------
function Y = doublepass(b,a,X)
% This internal function return the Y filtered vector after a zero-lag
% double pass iir filter on the original X vector.
temp = tofilt(b,a,X);
temp = temp(end:-1:1,:);
temp = tofilt(b,a,temp);
Y = temp(end:-1:1,:);
%--------------------------------------------------------------------------
function Y = tofilt(b,a,X)
% This internal function apply the second-order recursive (infinite impulse
% response, IIR) filter.
Y = X;
for i = 3:1:length(Y)
    Y(i,:) = (b(1).*X(i,:) + b(2).*X(i-1,:) + b(3).*X(i-2,:) + a(2).*Y(i-1,:) + a(3).*Y(i-2,:)) / a(1);
end