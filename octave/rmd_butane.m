clear

temp0 = 4.0;
tau = 0.01;
dt  = 0.002;

bondlength = 0.4;
bondconstant = 2000.0;
bondangle = 1.9;
angleconstant = 400.0;
torsionparam = [15.5000,  20.3050, -21.9170, -5.1150,  43.8340, -52.6070];

molsim('load', 'xyz', 'butane.xyz');
molsim('load', 'top', 'butane.top');

molsim('set','timestep', dt);
molsim('set', 'temperature', temp0);
molsim('set', 'exclusion', 'molecule');

molsim('sample', 'mvacf', 100, 5.0);
molsim('sample', 'radial', 100, 100, 'C');

tic
for n=1:1000

  molsim('reset')

  molsim('calcforce', 'lj', 'CC', 2.5, 1.0, 1.0);
  molsim('calcforce', 'bond', 0, bondlength, bondconstant);
  molsim('calcforce', 'angle', 0, bondangle, angleconstant);
  molsim('calcforce', 'torsion', 0, torsionparam);

  molsim('integrate', 'leapfrog')

  molsim('thermostate', 'C', temp0, tau);

  #molsim('sample', 'do');
  
  if rem(n,100) == 0
    printf("\r %d ", n );
    fflush(stdout);
  endif
  
endfor
toc


