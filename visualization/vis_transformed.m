clc;
Data = readtable('svm.csv');
x=log(Data{:,4});
%x=boxcox(2,Data{:,4});
%x=log2(Data{:,4});
[f,xi]=ksdensity(x);

figure
plot(xi,f);

xlabel('value')
ylabel('kernel density estimate')
title('kernel density estimate of the distribution of the values for the transformed svm data')