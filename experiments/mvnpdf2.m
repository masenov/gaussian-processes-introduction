function pdf = mvnpdf2(x,mu,sigma)
  [d,p] = size(x);
  % mu can be a scalar, a 1xp vector or a nxp matrix
  if nargin == 1, mu = 0; end
  if all(size(mu) == [1,p]), mu = repmat(mu,[d,1]); end
  if nargin < 3
    pdf = (2*pi)^(-p/2) * exp(-sumsqr(x-mu,2)/2);
  else
    r = chol(sigma);
    pdf = (2*pi)^(-p/2) * exp(-sumsqr((x-mu)/r,2)/2) / prod(diag(r));
  end