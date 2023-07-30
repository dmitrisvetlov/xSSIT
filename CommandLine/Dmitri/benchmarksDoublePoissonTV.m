clear all

M = 5;
Ns = 6;
Nsim = 5;
Nar = floor(logspace(1,2,Ns));
timeConstruction = nan*ones(Nsim,Ns);
timeode15s=nan*ones(Nsim,Ns);
t = 2;

k10 = 5;
k11 = 2;
om1 = 1;
k20 = 7;
k21 = 3;
om2 = 1;

k1 = @(t)k10+k11*sin(om1*t);
k2 = @(t)k20+k21*sin(om2*t);

g1 = 1;
g2 = 0.5;

for n = length(Nar):-1:1
    N = [Nar(n), Nar(n)];  % Projection size
        
    for j = 1:Nsim
        tic;
        A0 = zeros(prod(N+1),prod(N+1));
        A1 = zeros(prod(N+1),prod(N+1));
        A2 = zeros(prod(N+1),prod(N+1));
        C1 = zeros(N(1)+1,prod(N+1));
        C2 = zeros(N(2)+1,prod(N+1));
        
        for i1 = 0:N(1)
            for i2 = 0:N(2)
                k = i1*(N(2)+1)+i2+1;
                if i1>0
                    A0(k,k) = A0(k,k)-g1*i1;
                    A0(k-(N(2)+1),k) = g1*i1;
                end
                if i2>0
                    A0(k,k) = A0(k,k)-g2*i2;
                    A0(k-1,k) = g2*i2;
                end
                if i1<N(1)
                    A1(k,k) = A1(k,k)-1;
                    A1(k+(N(2)+1),k) = 1;
                end
                if i2<N(2)
                    A2(k,k) = A2(k,k)-1;
                    A2(k+1,k) = 1;
                end
                C1(i1+1,k) = 1;
                C2(i2+1,k) = 1;
            end % for i2 = 0:N(2) ...
        end % for i1 = 0:N(1) ...
        
        A0 = sparse(A0);
        A1 = sparse(A1);
        A2 = sparse(A2);
        C1 = sparse(C1);
        C2 = sparse(C2);

        P0 = zeros(prod(N+1),1); 
        P0(1) = 1;    

        A = @(t,x)A0+k1(t)*A1+k2(t)*A2;
        Apat = A(0,rand(size(P0)))~=0;

        timeConstruction(j,n) = toc;

        % ODE Calculation

        ode_opts = odeset('Jacobian', A, 'Vectorized','on','JPattern',Apat,...
            'relTol',1e-8, 'absTol', 1e-10,'NonNegative',true);
        rhs = @(t,x)A(t,x)*x;
        tic
        [tExport, yout] = ode15s(rhs, [0,t/2,t], P0, ode_opts);
        timeode15s(j,n) = toc;
        
        Pt = yout(end,:)';
        
        P1t = C1*Pt;
        P2t = C2*Pt;
    end % for j = 1:Nsim ...
end % for n = length(Nar):-1:1 ...

%% Comparisons
% Compare time
figure(1)
loglog(Nar,mean(timeode15s),'-^',Nar,mean(timeConstruction),'-+');
legend('ode15s','array-construction')
xlabel('Number of states')
ylabel('Computational time')
set(gca,'fontsize',15)

writematrix(mean(timeode15s), 'timeode15s_TV.m.txt');
writematrix(mean(timeConstruction), 'timeConstruction_TV.m.txt');