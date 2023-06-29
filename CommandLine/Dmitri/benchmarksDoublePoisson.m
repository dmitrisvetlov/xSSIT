clear all
%% Random Matrix
M = 5;
Ns = 6;
Nsim = 5;
Nar = floor(logspace(1,2,Ns));
timeConstruction = nan*ones(Nsim,Ns);
timeExpm =nan*ones(Nsim,Ns);
timeExpokit =nan*ones(Nsim,Ns);
diff_expm_expokit =nan*ones(Nsim,Ns);
diff_expokit_ode23s=nan*ones(Nsim,Ns);
diff_expm_ode23s=nan*ones(Nsim,Ns);
timeODE23s=nan*ones(Nsim,Ns);
t = 2;

k1 = 10;
k2 = 13;
g1 = 1;
g2 = 0.5;

for n = length(Nar):-1:1
    for j = 1:Nsim
        N = [Nar(n), Nar(n)];  % Projection size
        A = zeros(prod(N+1),prod(N+1));
        C1 = zeros(N(1)+1,prod(N+1));
        C2 = zeros(N(2)+1,prod(N+1));
        tic
        for i1 = 0:N(1)
            for i2 = 0:N(2)
                k = i1*(N(2)+1)+i2+1;
                if i1>0
                    A(k,k) = A(k,k)-g1*i1;
                    A(k-(N(2)+1),k) = g1*i1;
                end
                if i2>0
                    A(k,k) = A(k,k)-g2*i2;
                    A(k-1,k) = g2*i2;
                end
                if i1<N(1)
                    A(k,k) = A(k,k)-k1;
                    A(k+(N(2)+1),k) = k1;
                end
                if i2<N(2)
                    A(k,k) = A(k,k)-k2;
                    A(k+1,k) = k2;
                end
                C1(i1+1,k) = 1;
                C2(i2+1,k) = 1;
            end % for i2 = 0:N(2) ...
        end % for i1 = 0:N(1) ...
        timeConstruction(j,n) = toc;

        P0 = zeros(prod(N+1),1); 
        P0(1) = 1;

        %% Expm calculation
        tic
        expAt_P = expm(A*t)*P0;
        timeExpm(j,n) = toc;

        %% Expokit Calculation
        m = 15;
        tryAgain=1;
        %     fspTol = fspErrorCondition.fspTol;
        %     nSinks = fspErrorCondition.nSinks;
        fspErrorCondition.tInit = 0;
        tic
        while tryAgain==1
            [~, ~, ~, tExport, solutionsNow, ~, tryAgain, te, PfExpokit] = ssit.fsp_ode_solvers.mexpv_modified_2(t, ...
                A, P0, 1e-8, m,...
                [], [0,t], 1e-3,[], 0, fspErrorCondition);
            if tryAgain==0;break;end
            if m>300
                warning('Expokit expansion truncated at 300');
                [~, ~, ~, tExport, solutionsNow, ~, tryAgain, te, PfExpokit] = ssit.fsp_ode_solvers.mexpv_modified_2(tOut(end), jac, initSolution, fspTol/1e5, m,...
                    [], tOut, fspTol, [length(initSolution)-nSinks+1:length(initSolution)], tStart, fspErrorCondition);
            end
            m=m+5;
        end
        timeExpokit(j,n) = toc;

        %% ODE Calculation
        ode_opts = odeset('Jacobian', A, 'Vectorized','on','JPattern',A~=0,'relTol',1e-8, 'absTol', 1e-10);
        rhs = @(t,x)A*x;
        tic
        [tExport, yout] =  ode15s(rhs, [0,t/2,t], P0);
        timeODE23s(j,n) = toc;

        diff_expm_expokit(j,n) = sum(abs(PfExpokit-expAt_P));
        diff_expm_ode23s(j,n) = sum(abs(expAt_P-yout(end,:)'));
        diff_expokit_ode23s(j,n) = sum(abs(PfExpokit-yout(end,:)'));
    end % for j = 1:Nsim ...
end % for n = length(Nar):-1:1 ...

%% Comparisons
% Compare time
figure(2)
subplot(2,1,1);
loglog(Nar,mean(timeExpm),'-s',Nar,mean(timeExpokit),...
    '-o',Nar,mean(timeODE23s),'-^',Nar,mean(timeConstruction),'-+');
legend('expm','expokit','ode23s')
xlabel('Number of states')
ylabel('Computational time')
set(gca,'fontsize',15)

% Compare results
subplot(2,1,2);
plot(Nar,mean(diff_expm_expokit),'-o',Nar,mean(diff_expm_ode23s),'-s',Nar,mean(diff_expokit_ode23s),'-^')
legend('diff_expm_expokit','diff_expm_ode23s','diff_expokit_ode23s')