/**************************************
 * 
 **************************************/

void load_xyz(const char xyzfile[]);
void load_top(const char topfile[]);
void save_xyz(const char filename[]);

void free_memory(void);

void reset_iteration(void);

void force_lj(const char *types, float *ljparam);
void force_coulomb(float cf);

void integrate_leapfrog(void);

void thermostat_nh(float temp0, float thermostatmass);
void reset_momentum(int resetfreq);

void get_pressure(double *press);
void get_energies(double *energies);
