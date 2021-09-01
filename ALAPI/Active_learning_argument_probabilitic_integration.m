function [AK_MCS_Result] = Active_learning_argument_probabilitic_integration(Xinput,Pinput,Per_Fun, LF)
%ACTIVE_LEARNING_ARGUMENT_PROBABILITIC_INTEGRATION 此处显示有关此函数的摘要
%   此处显示详细说明

%% Step 1: generate a Monte Carlo population of n_mcs points
% the number of samples used in MCS
MCS.Size = 1E5;

dim = Xinput.Dim + Pinput.Dim;
% generate samples for the parameters: Theta
Usample = rand(MCS.Size,dim);
for i = Xinput.Dim+1:dim
    MCS.Sample(:,i) = unifinv(unifcdf(Usample(:,i)),Pinput.Bound(1,i-Xinput.Dim),Pinput.Bound(2,i-Xinput.Dim));
end

% secondly tansform the samples  to the physical space: X
for j = 1:Xinput.Dim
    switch Xinput.Dist{j}
        case 'Normal'
            MCS.Sample(:,j) = norminv(unifcdf(Usample(:,j)),MCS.Sample(:,Xinput.Dim+2*j-1),MCS.Sample(:,Xinput.Dim+2*j));
    end
end

Pdf = @(x) normpdf(x(:,1),x(:,Xinput.Dim+1),x(:,Xinput.Dim+2))...
    .*normpdf(x(:,2),x(:,Xinput.Dim+3),x(:,Xinput.Dim+4))...
    .*unifpdf(x(:,Xinput.Dim+1),Pinput.Bound(1,1),Pinput.Bound(2,1))...
    .*unifpdf(x(:,Xinput.Dim+2),Pinput.Bound(1,2),Pinput.Bound(2,2))...
    .*unifpdf(x(:,Xinput.Dim+3),Pinput.Bound(1,3),Pinput.Bound(2,3))...
    .*unifpdf(x(:,Xinput.Dim+4),Pinput.Bound(1,4),Pinput.Bound(2,4));

%% Step 2: define the initial design of experiments (DoE)
% the size of the initial DOE
DoE.Size = 12;
% random selection from the MCS population, e.g. the first DoE.Size samples
Xini = MCS.Sample(1:DoE.Size,:);
% evaluate the performance function
Yini = Per_Fun(Xini);





Max_AX = 500;

% delayed judgement stragety

Delayed_num = 3;


% theta0 = 1.*ones(1,dim);
% lob = 1E-3.*ones(1,dim);
% upb = 1e3.*ones(1,dim);


for j = 1:Max_AX
    %train or update the Gaussian process regression model
    GPRmodel = fitrgp(Xini,Yini,'BasisFunction','constant','KernelFunction','ardsquaredexponential','ConstantSigma',true,'Sigma',1e-12,'SigmaLowerBound',eps, 'verbose', 0,'Standardize',0);
    
    % predictor
    [ymu,ysd] = predict(GPRmodel,MCS.Sample);
    
    %     [dmodel, perf] = dacefit(Xini, Yini, @regpoly1, @corrgauss, theta0,lob, upb);
    %     [YX,MSE] = predictor(MCS.Sample, dmodel);
    %     ymu = YX;
    %     ysd = sqrt(MSE);
    %
    
    
    % failure proability
    pf_Kriging = mean(normcdf(-ymu./ysd));
    
    %  identify the best next point to be added
    switch LF
        case 'U'
            Learning_Fun_U = @(mu,st) abs(mu)./st;
            U = Learning_Fun_U(ymu,ysd);
            AX = MCS.Sample(find(U==min(U)),:); % added X
        case 'UPVC'
            Learning_Fun_UPVC = @(mu,st) normcdf(-mu./st).*normcdf(mu./st);
            UPVC = Learning_Fun_UPVC(ymu,ysd);
            WUPVC = sqrt(UPVC).*Pdf(MCS.Sample);
            AX = MCS.Sample(find(WUPVC==max(WUPVC)),:); % added X
        otherwise
            warning('Check your learning function!')
    end
    
    % check the stopping condition
    switch LF
        case 'U'
            if min(U) >= 2
                break;
            else
                min(U)
                AY = Per_Fun(AX);
                Xini = [Xini;AX];
                Yini = [Yini;AY];
                fprintf('AK-MCS:%d samples added \n',j)
            end
            
        case 'UPVC'
            COV(j) = mean(sqrt(UPVC))./pf_Kriging;
            COV(max(end-Delayed_num+1,1):end)
            if  (j>=Delayed_num) && (sum(COV(end-Delayed_num+1:end)<=0.02) == Delayed_num)
                break;
            else
                AY = Per_Fun(AX);
                Xini = [Xini;AX];
                Yini = [Yini;AY];
                fprintf('AK-MCS:%d samples added \n',j)
                %             end
                
            end
    end
    
end

% CoV = sqrt((1-pf_Kriging)./(pf_Kriging*MCS.Size));
% sprintf('The C.O.V of the estimated Pf: %f',CoV)

AK_MCS_Result.FunCalls = DoE.Size + j - 1;
% AK_MCS_Result.X = Xini;

AK_MCS_Result.UCOV = COV;


% constand component


AK_MCS_Result.Pf0 = mean(normcdf(-ymu./ysd));
AK_MCS_Result.Pf0_UCOV = mean(sqrt(normcdf(-ymu./ysd).*normcdf(ymu./ysd)))./AK_MCS_Result.Pf0;




% u1
u1 = Pinput.Bound(1,1):(Pinput.Bound(2,1)-Pinput.Bound(1,1))./29:Pinput.Bound(2,1);
for i = 1:length(u1)
    sample_u1 = MCS.Sample;
    sample_u1(:,3) = u1(i);
    sample_u1(:,1) = normrnd(sample_u1(:,3),sample_u1(:,4),MCS.Size,1);
    [ymu1,ysd1] = predict(GPRmodel,sample_u1);
    Pf_1(i) = mean(normcdf(-ymu1./ysd1));%-normcdf(-ymu./ysd));
    Pf_1_UCOV(i) = mean(sqrt(normcdf(-ymu1./ysd1).*normcdf(ymu1./ysd1)))./Pf_1(i);
end
AK_MCS_Result.u1 = u1;
AK_MCS_Result.Pf1 = Pf_1;
AK_MCS_Result.Pf1_UCOV = Pf_1_UCOV;
 
% s1
s1 = Pinput.Bound(1,2):(Pinput.Bound(2,2)-Pinput.Bound(1,2))./29:Pinput.Bound(2,2);
for i = 1:length(s1)
    sample_u1 = MCS.Sample;
    sample_u1(:,4) = s1(i);
    sample_u1(:,1) = normrnd(sample_u1(:,3),sample_u1(:,4),MCS.Size,1);
    [ymu2,ysd2] = predict(GPRmodel,sample_u1);
    Pf_2(i) = mean(normcdf(-ymu2./ysd2));
    Pf_2_UCOV(i) = mean(sqrt(normcdf(-ymu2./ysd2).*normcdf(ymu2./ysd2)))./Pf_2(i);
end
AK_MCS_Result.s1 = s1;
AK_MCS_Result.Pf2 = Pf_2;
AK_MCS_Result.Pf2_UCOV = Pf_2_UCOV;

% u2
u2 = Pinput.Bound(1,3):(Pinput.Bound(2,3)-Pinput.Bound(1,3))./29:Pinput.Bound(2,3);
for i = 1:length(u2)
    sample_u1 = MCS.Sample;
    sample_u1(:,5) = u2(i);
    sample_u1(:,2) = normrnd(sample_u1(:,5),sample_u1(:,6),MCS.Size,1);
    [ymu3,ysd3] = predict(GPRmodel,sample_u1);
    Pf_3(i) = mean(normcdf(-ymu3./ysd3));
    Pf_3_UCOV(i) = mean(sqrt(normcdf(-ymu3./ysd3).*normcdf(ymu3./ysd3)))./Pf_3(i);
end
AK_MCS_Result.u2 = u2;
AK_MCS_Result.Pf3 = Pf_3;
AK_MCS_Result.Pf3_UCOV = Pf_3_UCOV;


% s2
s2 = Pinput.Bound(1,4):(Pinput.Bound(2,4)-Pinput.Bound(1,4))./29:Pinput.Bound(2,4);
for i = 1:length(s2)
    sample_u1 = MCS.Sample;
    sample_u1(:,6) = s2(i);
    sample_u1(:,2) = normrnd(sample_u1(:,5),sample_u1(:,6),MCS.Size,1);
    [ymu4,ysd4] = predict(GPRmodel,sample_u1);
    Pf_4(i) = mean(normcdf(-ymu4./ysd4));
    Pf_4_UCOV(i) = mean(sqrt(normcdf(-ymu4./ysd4).*normcdf(ymu4./ysd4)))./Pf_4(i);
end
AK_MCS_Result.s2 = s2;
AK_MCS_Result.Pf4 = Pf_4;
AK_MCS_Result.Pf4_UCOV = Pf_4_UCOV;
% 
for i = 1:length(u1)
    for j = 1:length(u2)
        (i-1)*length(u2)+j
        sample_u1 = MCS.Sample;
        sample_u1(:,3) = u1(i);        
        sample_u1(:,5) = u2(j);
        sample_u1(:,1) = normrnd(sample_u1(:,3),sample_u1(:,4),MCS.Size,1);
        sample_u1(:,2) = normrnd(sample_u1(:,5),sample_u1(:,6),MCS.Size,1);
        [ymu13,ysd13] = predict(GPRmodel,sample_u1);
        Pf_13(i,j) = mean(normcdf(-ymu13./ysd13));
        Pf_13_UCOV(i,j) = mean(sqrt(normcdf(-ymu13./ysd13).*normcdf(ymu13./ysd13)))./Pf_13(i,j);
    end
end

AK_MCS_Result.Pf13 = Pf_13;
AK_MCS_Result.Pf13_UCOV = Pf_13_UCOV;


end


% function [Pdf] = joint_pdf(v,Xinput,Pinput)
%    
% 
% end
