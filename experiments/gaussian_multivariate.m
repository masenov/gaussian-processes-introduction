function y = gaussian_multivariate( x,mu,sigma )
d = length(mu);
x_dim = size(x);
dim = x_dim(1);
y = zeros(dim,1);
for i = 1:dim
    y(i) = exp(-(x(i,:)-mu)*inv(sigma)*(x(i,:)-mu)'/2)/((2*pi)^(d/2)*det(sigma)^(1/2));
end

