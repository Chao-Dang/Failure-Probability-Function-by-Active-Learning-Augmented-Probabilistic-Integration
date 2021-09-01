function [IASS_Result] = Imprecise_augmented_stochastic_simulation(Xinput,Pinput,Per_Fun,num_mcs)
%IMPRECISE_AUGMENTED_STOCHASTIC_SIMULATION 此处显示有关此函数的摘要
%   此处显示详细说明


Usample = [unifrnd(Pinput.Bound(1,1),Pinput.Bound(2,1),num_mcs,1),unifrnd(Pinput.Bound(1,2),Pinput.Bound(2,2),num_mcs,1),unifrnd(Pinput.Bound(1,3),Pinput.Bound(2,3),num_mcs,1),unifrnd(Pinput.Bound(1,4),Pinput.Bound(2,4),num_mcs,1)];
for i = 1:Xinput.Dim
    switch Xinput.Dist{i}
        case 'Normal'
            Xsample(:,i) = normrnd(Usample(:,2*i-1),Usample(:,2*i),num_mcs,1);
    end
end

IF0 = Per_Fun(Xsample)<0;

Pf_0 = mean(IF0);
Pf_0_COV = std(IF0)./sqrt(num_mcs)./Pf_0;

IASS_Result.Pf0 = Pf_0;
IASS_Result.Pf0_COV = Pf_0_COV;


% first-order HDMR components
% u1
u1 = Pinput.Bound(1,1):(Pinput.Bound(2,1)-Pinput.Bound(1,1))./29:Pinput.Bound(2,1);
for j = 1:length(u1)
    j
    Usample = [repmat(u1(j),num_mcs,1),unifrnd(Pinput.Bound(1,2),Pinput.Bound(2,2),num_mcs,1),unifrnd(Pinput.Bound(1,3),Pinput.Bound(2,3),num_mcs,1),unifrnd(Pinput.Bound(1,4),Pinput.Bound(2,4),num_mcs,1)];
    Xsample = [normrnd(Usample(:,1),Usample(:,2),num_mcs,1),normrnd(Usample(:,3),Usample(:,4),num_mcs,1)];
    IF1 = Per_Fun(Xsample)<0;
    Pf_1(j) = mean(IF1);
    Pf_1_COV(j) = std(IF1)./sqrt(num_mcs)./Pf_1(j);
end
IASS_Result.Pf1 = Pf_1;
IASS_Result.Pf1_COV = Pf_1_COV;



% s1
s1 = Pinput.Bound(1,2):(Pinput.Bound(2,2)-Pinput.Bound(1,2))./29:Pinput.Bound(2,2);
for j = 1:length(s1)
    j
    Usample = [unifrnd(Pinput.Bound(1,1),Pinput.Bound(2,1),num_mcs,1),repmat(s1(j),num_mcs,1),unifrnd(Pinput.Bound(1,3),Pinput.Bound(2,3),num_mcs,1),unifrnd(Pinput.Bound(1,4),Pinput.Bound(2,4),num_mcs,1)];
    Xsample = [normrnd(Usample(:,1),Usample(:,2),num_mcs,1),normrnd(Usample(:,3),Usample(:,4),num_mcs,1)];
    IF2 = Per_Fun(Xsample)<0;
    Pf_2(j) = mean(IF2);
    Pf_2_COV(j) = std(IF2)./sqrt(num_mcs)./Pf_2(j);
end
IASS_Result.Pf2 = Pf_2;
IASS_Result.Pf2_COV = Pf_2_COV;



% u2
u2 = Pinput.Bound(1,3):(Pinput.Bound(2,3)-Pinput.Bound(1,3))./29:Pinput.Bound(2,3);
for j = 1:length(u2)
    j
    Usample = [unifrnd(Pinput.Bound(1,1),Pinput.Bound(2,1),num_mcs,1),unifrnd(Pinput.Bound(1,2),Pinput.Bound(2,2),num_mcs,1),repmat(u2(j),num_mcs,1),unifrnd(Pinput.Bound(1,4),Pinput.Bound(2,4),num_mcs,1)];
    Xsample = [normrnd(Usample(:,1),Usample(:,2),num_mcs,1),normrnd(Usample(:,3),Usample(:,4),num_mcs,1)];
    IF3 = Per_Fun(Xsample)<0;
    Pf_3(j) = mean(IF3);
    Pf_3_COV(j) = std(IF3)./sqrt(num_mcs)./Pf_3(j);
end
IASS_Result.Pf3 = Pf_3;
IASS_Result.Pf3_COV = Pf_3_COV;



% s2
s2 = Pinput.Bound(1,4):(Pinput.Bound(2,4)-Pinput.Bound(1,4))./29:Pinput.Bound(2,4);
for j = 1:length(s2)
    j
    Usample = [unifrnd(Pinput.Bound(1,1),Pinput.Bound(2,1),num_mcs,1),unifrnd(Pinput.Bound(1,2),Pinput.Bound(2,2),num_mcs,1),unifrnd(Pinput.Bound(1,3),Pinput.Bound(2,3),num_mcs,1),repmat(s2(j),num_mcs,1)];
    Xsample = [normrnd(Usample(:,1),Usample(:,2),num_mcs,1),normrnd(Usample(:,3),Usample(:,4),num_mcs,1)];
    IF4 = Per_Fun(Xsample)<0;
    Pf_4(j) = mean(IF4);
    Pf_4_COV(j) = std(IF4)./sqrt(num_mcs)./Pf_4(j);
end
IASS_Result.Pf4 = Pf_4;
IASS_Result.Pf4_COV = Pf_4_COV;


% second-order component function
for i = 1:length(u1)
    for j = 1:length(u2)
        (i-1)*length(u2)+j
       Usample = [repmat(u1(i),num_mcs,1),unifrnd(Pinput.Bound(1,2),Pinput.Bound(2,2),num_mcs,1),repmat(u2(j),num_mcs,1),unifrnd(Pinput.Bound(1,4),Pinput.Bound(2,4),num_mcs,1)];
       Xsample = [normrnd(Usample(:,1),Usample(:,2),num_mcs,1),normrnd(Usample(:,3),Usample(:,4),num_mcs,1)];
      IF13 = Per_Fun(Xsample)<0;
      Pf_13(i,j) = mean(IF13);
      Pf_13_COV(i,j) = std(IF13)./sqrt(num_mcs)./Pf_13(i,j);
    end
end

IASS_Result.Pf13 = Pf_13;
IASS_Result.Pf13_COV = Pf_13_COV;

end

