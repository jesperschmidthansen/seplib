clear

cutoff = 2.5;
epsilon = 1.0;
sigma = 1.0;
dt = 0.0025;

temp0 = 2.0;
tau = 0.01;

molsim('load', 'xyz', 'start.xyz');

molsim('set', 'temperature', temp0);
molsim('set', 'timestep', dt);

npart = molsim('get', 'numbpart');

m = 1;

for n=1:50000

  molsim('reset');

  molsim('calcforce', 'lj', 'AA', cutoff, sigma, epsilon);

  molsim('integrate', 'leapfrog');

  molsim('thermostate', 'relax', 'A', temp0, tau); 

  if ( temp0 > 0.2 )
    temp0 = 0.9999 * temp0;
  end
  
  if rem(n,500)==0 

    energies(m,:) = molsim('get', 'energies');
    pressure(m) = molsim('get', 'pressure');
    t(m) = n*dt;

    m = m + 1;
    
    figure(1);
    plot(t, energies(:,1)/npart*2/3, ";temperature;", ...
	 t, energies(:,2)/npart, ";Pot. energy;", ...
	 t, pressure, ";pressure;");
    xlabel('time');
    
    figure(2);
    pos = molsim('get', 'positions');
    plot3(pos(:,1), pos(:,2),pos(:,3), 'o', 'markersize', 8, ...
	'markeredgecolor', 'k', 'markerfacecolor', 'b');

    pause(0.01)
   
  end

end
  

