% System
A = [1 1; 0 1];
B = [0; 1];

% Compute LQR controller
Q = diag([1 1]);
R = 0.1;
K = dlqr(A, B, Q, R);

% State constraints
Hx = [1 0; 0 1; -1 0; 0 -1];
hx = [10; 10; 10; 10];
X  = polytope(Hx,hx);

% Input constraints
Hu = [1; -1];
hu = [1; 1];
U  = polytope(Hu,hu);

% Closed-loop dynamics
Ak = A-B*K;
% State and input constraints
HH = [Hx;Hu*K]; hh = [hx;hu];

% Compute the maximal invariant set
i = 1;
O = polytope(HH,hh);
while 1
	Oprev = O;
	[F,f] = double(O);	
	% Compute the pre-set
	O = polytope([F;F*Ak],[f;f]);
	if O == Oprev, break; end
	
	fprintf('Iteration %i...\n', i)

	i = i + 1;
end

[F,f] = double(O);