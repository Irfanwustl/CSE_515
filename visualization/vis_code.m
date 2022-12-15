clc;
Data = readtable('lda.csv');
x=Data{:,4};

[f,xi]=ksdensity(x);

figure
plot(xi,f);
xlabel('value')
ylabel('kernel density estimate')
title('kernel density estimate of the distribution of the values for the lda')