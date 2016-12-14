function y = gaussian( x,mu,sigma )

y = exp(-(x-mu).^2/(2*sigma.^2))/(2*pi*sigma.^2)^(1/2);

end

