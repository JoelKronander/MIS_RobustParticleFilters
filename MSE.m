function [ MSE ] = MSE( true_x, est_x )

MSE = mean(mean((true_x-est_x).^2));

end

