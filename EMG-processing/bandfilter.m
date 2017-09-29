function Y = bandfilter(X,lpFc,hpFc,Fs,type,n)
%
%bandfilter design and apply band-pass iir filter coefficients for Butterworth
%or critically damped filters using the lpfilter and hpfilter functions.
%
%by Cédric Morio - May 2009
%Oxylane Research - Department of Movement Sciences
%
%   Y = bandfilter(X, lpFc, hpFc, Fs) Filters the X signal with a 4th order 
%   Butterworth zero-lag filter (2 passes of a 2nd order). Returns the Y
%   filtered signal. lpFc and hpFc are the cut-off frequency of the bandpass
%   filter and Fs the sampling frequency of the original signal.
%
%   Y = bandfilter(X, lpFc, hpFc, Fs, TYPE) Filters the X signal with a 4th 
%   order Butterworth zero-lag filter (1 two-passes of a 2nd order) if TYPE 
%   is 'butter' and filters the X vector with a 20th order critically-damped
%   filter zero-lag too (5 two-passes of a second order) if TYPE is 'damped'.
%   Returns the Y filtered signal.
%
%   Y = bandfilter(X, lpFc, hpFc, Fs, TYPE, N) Filters the X signal with a 
%   2nd order filter 'butter' or 'damped' with N passes. If the number of 
%   passes N is impair for a simple filter and if it is pair for a zero-lag
%   phase filter. Returns the Y filtered signal.
%
%   References:
%   [1] D.G.E. Robertson, J.J. Dowling. Design and responses of Butterworth and
%   critically damped digital filters. Journal of Electromyography and
%   Kinesiology, vol 13, 2003, pp566-573.
%   [2] D.A. Winter. Biomechanics and Motor Control of Human Movement. 2nd
%   edition, 1990, pp36-41.
%   [3] S.W. Smith. The scientist and engineer's guide to digital signal
%   processing. www.dspguide.com
%
%   See also LPFILTER, HPFILTER, BUTTER, FILTER, FILTFILT
    if nargin < 4
        error('not enough arguments when calling bandfilter.');
    elseif nargin > 6
        error('too much arguments when calling bandfilter.');
    elseif nargin == 4
        type = 'butter'; % type of filter is not specified
        n = 2; % number of passes is not specified
    elseif nargin == 5
        switch type
            case 'butter'
                n = 2;
            case 'damped'
                n = 10;
            otherwise
                error('in lpfilter the low-pass filter type must be ''butter'' or ''damped''.');
        end
    end
    if hpFc <= lpFc
        error('the lowpass frequency must be higher than the highpass frequency.');
    end
    if n < 1
        error('in lpfilter the number of passes n must be an positive integer.');
    end
    if lpFc <= 0 || hpFc <= 0 || Fs <= 0
        error('in lpfilter the cutoff frequency Fc and the sampling rate Fs must be positive.');
    end
    % realisation du filtrage
    % -----------------------
    F = lpfilter(X,hpFc,Fs,type,n);
    Y = hpfilter(F,lpFc,Fs,type,n);
