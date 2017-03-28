% this is used to convert dataset X to Xtil via Random Fourier Features

function Xtil = RRF(X,NoFeatures,sigma)
W = randn(NoFeatures,size(X,2));
Xtil = [cos(X*W'/sigma),sin(X*W'/sigma)]/sqrt(NoFeatures);
