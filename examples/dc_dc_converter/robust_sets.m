clear all
close all
clc

%%
A = [0.971356387900839 -0.009766890567613; 1.731870774203751 0.970462385352837];
B = [0.148778899882612; 0.180827260808426];
C = [0 1];

Q = diag([90, 1]);
R = 1;

ref = 5;
ss = [A - eye(2) B; C 0] \ [0; 0; ref];
xs = ss(1:2);
us = ss(3);

xub = [0.2; 7.0];
xlb = [0; 0];
uub = 1;
ulb = 0;

u_err_ub = 0.1;
u_err_lb = -u_err_ub;

[K,Qf,~] = dlqr(A,B,Q,R);
K = -K;

Acl = A + B*K;

W = B * Polyhedron([1; -1], [u_err_ub; -u_err_lb]);

%%
F_inf = Polyhedron();
AcliW = W;
i = 1;
while 1
    disp(['Iteration ', num2str(i)])
    F_next = plus(F_inf,AcliW);
    if isequal(F_next, F_inf) || i >= 100
        break;
    end
    F_inf = F_next;
    AcliW = Acl * AcliW;
    i = i + 1;
end

%%
X = Polyhedron([eye(2); -eye(2)], [xub; -xlb]);
X_tight = minus(X, F_inf);

U = Polyhedron([eye(1); -eye(1)], [uub; -ulb]);
U_tight = minus(U, K * F_inf);

%%
X_tight_shift = X_tight - xs;
U_tight_shift = U_tight - us;
Xf_tight_shift = Polyhedron([X_tight_shift.A; U_tight_shift.A * K], [X_tight_shift.b; U_tight_shift.b]);
i = 1;
while 1
    disp(['Iteration ', num2str(i)])
    prev_Xf = Xf_tight_shift;
    pre_Xf = Polyhedron(Xf_tight_shift.A * Acl,Xf_tight_shift.b);
    Xf_tight_shift = intersect(Xf_tight_shift, pre_Xf);
    if isequal(prev_Xf, Xf_tight_shift)
        break
    end
    i = i + 1;
end

%%
F_inf_A = F_inf.A; F_inf_b = F_inf.b;
X_tight_A = X_tight.A; X_tight_b = X_tight.b;
U_tight_A = U_tight.A; U_tight_b = U_tight.b;
save('robust_sets.mat', 'F_inf_A', 'F_inf_b', 'X_tight_A', 'X_tight_b', 'U_tight_A', 'U_tight_b');
