clear all

Ns = 6;
Nsim = 5;
Nar = floor(logspace(1,2,Ns)); % 1,2
timeConstruction = nan*ones(Nsim,Ns);
timeode15s=nan*ones(Nsim,Ns);
err1=nan*ones(Nsim,Ns);
err2=nan*ones(Nsim,Ns);
t = 2;

k10 = 5;
k11 = 2;
om1 = 1;
k20 = 5; %7
k21 = 2; %3
om2 = 1; %1

k1 = @(t)k10+k11*sin(om1*t);
k2 = @(t)k20+k21*sin(om2*t);

g1 = 1;
g2 = 1; %0.5

figure(1)
hold on

for n = 1:length(Nar) % length(Nar):-1:1
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
        %Apat = A(0,rand(size(P0)))~=0;

        timeConstruction(j,n) = toc;

        % ODE Calculation

        ode_opts = odeset('Jacobian', A, 'Vectorized','on',...
            'relTol',1e-8, 'absTol', 1e-10,'NonNegative',true);
        %ode_opts2 = odeset('JPattern',Apat);
        %ode_opts = odeset(ode_opts, ode_opts2);
        rhs = @(t,x)A(t,x)*x;
        tic
        [tExport, yout] = ode15s(rhs, [0,t/2,t], P0, ode_opts);
        timeode15s(j,n) = toc;
        
        Pt = yout(end,:)';
        
        P1t = C1*Pt;
        P2t = C2*Pt;

        %% Check results

        lam1 = k10/g1*(1-exp(-g1*t)) + ...
            (k11*om1*exp(-g1*t))/(g1^2 + om1^2) - (k11*(om1*cos(om1*t) - g1*sin(om1*t)))/(g1^2 + om1^2);
        y1 = poisspdf([0:N(1)]',lam1);
        err1(j,n) = sum(abs(y1-P1t));
        
        lam2 = k20/g2*(1-exp(-g2*t)) + ...
            (k21*om2*exp(-g2*t))/(g2^2 + om2^2) - (k21*(om2*cos(om2*t) - g2*sin(om2*t)))/(g2^2 + om2^2);
        y2 = poisspdf([0:N(2)]',lam2);
        err2(j,n) = sum(abs(y2-P2t));

        if j==1
            plot([0:N(1)]',y1,'r--')
            plot([0:N(2)]',y2,'b--')
        end
    end % for j = 1:Nsim ...
end % for n = length(Nar):-1:1 ...

hold off

%% Comparisons
% Compare time
figure(2)
loglog(Nar,mean(timeode15s),'-^',Nar,mean(timeConstruction),'-+');
legend('ode15s','array-construction')
xlabel('Number of states')
ylabel('Computational time')
set(gca,'fontsize',15)

writematrix(mean(timeode15s), 'timeode15s_TV.m.txt');
writematrix(mean(timeConstruction), 'timeConstruction_TV.m.txt');

figure(3)
loglog(Nar,mean(err1),'-^',Nar,mean(err2),'-+');

writematrix(mean(err1), 'err1_TV.m.txt');
writematrix(mean(err2), 'err2_TV.m.txt');