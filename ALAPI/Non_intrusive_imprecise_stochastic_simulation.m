function [NISS_Result] = Non_intrusive_imprecise_stochastic_simulation(Xinput,Pinput,Per_Fun)
%NON_INTRUSIVE_STOCHASTIC_SIMULATION This is the "AK-GEMCS-RS-HDMR"  method
% described in NISS part 2
%

%% Step 1: generate a Monte Carlo population of n_mcs points

% the number of samples used in MCS
MCS.Size = 1E5;

% genertate simple random samples
U = rand(MCS.Size,Xinput.Dim+Pinput.Dim);
for i = 1:Pinput.Dim
    switch Pinput.Dist{i}
        case 'Uniform'
            Psample(:,i) = unifinv(U(:,Xinput.Dim+i),Pinput.Bound(1,i), Pinput.Bound(2,i));
        case 'Normal'
        otherwise
            disp('Check your distribution type!')
    end
end

for j = 1:Xinput.Dim
    switch Xinput.Dist{j}
        case 'Normal'
            MCS.Sample(:,j) = norminv(U(:,j),Psample(:,2*j-1),Psample(:,2*j));
        case 'Logormal'
            %                     MCS.Sample(:,j) = lognrnd(Xinput.Para(1,j),Xinput.Para(2,j),MCS.Size,1);
        otherwise
            disp('Check your distribution type!')
    end
end

%% Step 2: define the initial design of experiments (DoE)
% the size of the initial DOE
DoE.Size = 12;
% random selection from the MCS population, e.g. the first DoE.Size samples
X0 = MCS.Sample(1:DoE.Size,:);
% evaluate the performance function
Y0 = Per_Fun(X0);

%%  Step 3: train the initial Kriging model
Max_AX = 500;

for j = 1:Max_AX
    %train or update the Gaussian process regression model
    GPRmodel = fitrgp(X0,Y0,'BasisFunction','constant','KernelFunction','ardsquaredexponential','ConstantSigma',true,'Sigma',1e-12,'SigmaLowerBound',eps, 'verbose', 0,'Standardize',0);
    
    %% Step 4: Kriging Predictor
    % predictor
    [ymu,ysd] = predict(GPRmodel,MCS.Sample);
    
    % failure probability
    pf_Kriging = length(find(ymu<=0))./MCS.Size;
    %% Step 5: identify the best next point to be added
    % learning function U
    Learning_Fun_U =@(Y,Std) abs(Y)./Std;
    
    U = Learning_Fun_U(ymu,ysd);
    AX = MCS.Sample(find(U==min(U)),:); % added X
    %% Step 6: check the stopping criteria
    if min(U) >= 2
        break;
    else
        min(U)
        AY = Per_Fun(AX);
        X0 = [X0;AX];
        Y0 = [Y0;AY];
        fprintf('AK-MCS:%d samples added \n',j)
    end
      
end

NISS_Result.FunCalls = DoE.Size + j - 1;

% constand component
NISS_Result.Pf0 = mean(ymu<0);
NISS_Result.Pf0_COV = std(ymu<0)./NISS_Result.Pf0./sqrt(MCS.Size);

% first-order components
Xpdf = @(x,p) normpdf(x(:,1),p(:,1),p(:,2)).*normpdf(x(:,2),p(:,3),p(:,4));
Theta_k = @(p,k) [Psample(:,1:k-1),repmat(p,MCS.Size,1),Psample(:,k+1:4)];


FPF_est = @(p,k) mean((ymu<0).*(Xpdf(MCS.Sample,Theta_k(p,k))./Xpdf(MCS.Sample,Psample)));
FPF_std = @(p,k) std((ymu<0).*(Xpdf(MCS.Sample,Theta_k(p,k))./Xpdf(MCS.Sample,Psample)))./sqrt(MCS.Size);

% u1
u1 = Pinput.Bound(1,1):(Pinput.Bound(2,1)-Pinput.Bound(1,1))./29:Pinput.Bound(2,1);
for i = 1:length(u1)
    Pf_1(i) = FPF_est(u1(i),1);
    Pf_1_STD(i) = FPF_std(u1(i),1);
end
NISS_Result.u1 = u1;
NISS_Result.Pf1 = Pf_1;
NISS_Result.Pf1_COV = Pf_1_STD./Pf_1;

% s1
s1 = Pinput.Bound(1,2):(Pinput.Bound(2,2)-Pinput.Bound(1,2))./29:Pinput.Bound(2,2);
for i = 1:length(s1)
    Pf_2(i) = FPF_est(s1(i),2);
    Pf_2_STD(i) = FPF_std(s1(i),2);
end
NISS_Result.s1 = s1;
NISS_Result.Pf2 = Pf_2;
NISS_Result.Pf2_COV = Pf_2_STD./Pf_2;

% u2
u2 = Pinput.Bound(1,3):(Pinput.Bound(2,3)-Pinput.Bound(1,3))./29:Pinput.Bound(2,3);
for i = 1:length(u2)
    Pf_3(i) = FPF_est(u2(i),3);
    Pf_3_STD(i) = FPF_std(u2(i),3);
end
NISS_Result.u2 = u2;
NISS_Result.Pf3 = Pf_3;
NISS_Result.Pf3_COV = Pf_3_STD./Pf_3;

% s2
s2 = Pinput.Bound(1,4):(Pinput.Bound(2,4)-Pinput.Bound(1,4))./29:Pinput.Bound(2,4);
for i = 1:length(s2)
    Pf_4(i) = FPF_est(s2(i),4);
    Pf_4_STD(i) = FPF_std(s2(i),4);
end
NISS_Result.s2 = s2;
NISS_Result.Pf4 = Pf_4;
NISS_Result.Pf4_COV = Pf_4_STD./Pf_4;

end


