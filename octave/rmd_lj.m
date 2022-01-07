clear

cutoff = 2.5;
epsilon = 1.0;
sigma = 1.0;

molsim('load', 'xyz', 'start.xyz');

for n=1:10000

  molsim('reset');
  molsim('calcforce', 'lj', 'AA', cutoff, sigma, epsilon);
  molsim('integrate', 'leapfrog');

  if rem(n, 100)==0
    molsim('print');
  end

end

molsim('clear');

