set points;

param p;
param a_max;

param arr {points};
param dep {points};
param freq {points};

var alpha {1..a_max} >= 0;
var beta {1..a_max} <= 0;

minimize z:	sum {i in points} freq[i]*(p*max(0,dep[i] - (alpha[if arr[i] > 0 then arr[i] else 1]+beta[if arr[i] > 0 then arr[i] else 1]*arr[i])) + (1-p)*max(0,alpha[if arr[i] > 0 then arr[i] else 1]+beta[if arr[i] > 0 then arr[i] else 1]*arr[i] - dep[i]));

subject to c1 {k in 1..(a_max-1)}:
	beta[k+1] <= beta[k];

subject to c2 {k in 1..(a_max-1)}:
	alpha[k] + beta[k]*k = alpha[k+1] + beta[k+1]*k