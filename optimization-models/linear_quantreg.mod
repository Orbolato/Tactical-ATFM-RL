set points;

param p;
param a_max;

param arr {points};
param dep {points};
param freq {points};

var alpha {1..a_max};
var beta {1..a_max} <= 0;
var z_n {points};

minimize z:	sum {i in points} freq[i]*z_n[i];

subject to c1 {i in points}:
	z_n[i] >= dep[i] - (alpha[if arr[i] > 0 then arr[i] else 1]+beta[if arr[i] > 0 then arr[i] else 1]*arr[i]);

subject to c2 {i in points}:
	z_n[i] >= (1-p)/p*(alpha[if arr[i] > 0 then arr[i] else 1]+beta[if arr[i] > 0 then arr[i] else 1]*arr[i] - dep[i]);
	
subject to c3 {k in 1..(a_max-1)}:
	beta[k+1] <= beta[k];

subject to c4 {k in 1..(a_max-1)}:
	alpha[k] + beta[k]*k = alpha[k+1] + beta[k+1]*k