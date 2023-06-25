#########################################
##
## Scanity check
##
##########################################

clear

system('export LD_LIBRARY_PATH=.');

cutoff = 2.5; epsilon = 1.0; sigma = 1.0; aw = 1.0;

cmolsim('load', 'xyz', 'start_singleAN1000.xyz');
cmolsim('set', 'resetmomentum', 100);

for n=1:100
	
	cmolsim('reset');

	cmolsim('calcforce', 'lj', 'AA', cutoff, sigma, epsilon, aw);

	cmolsim('integrate', 'leapfrog');

	cmolsim('thermostat', 'nosehoover', 'A', 2.0, 0.1);

	if rem(n,10)==0 
		pressure = cmolsim('get', 'pressure');
		%energies = cmolsim('get', 'energies');
		printf("%d %f\n", n, pressure(1));
		%printf("%d %f %f %f\n", n, pressure(1), energies(1), energies(2));
		fflush(stdout);
	endif
	
end

cmolsim('save', 'test.xyz');

% Free memory allocated
cmolsim('clear');
