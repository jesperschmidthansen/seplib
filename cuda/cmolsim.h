/**************************************
 * 
 **************************************/

void load_xyz(const char xyzfile[]);
void load_top(const char topfile[]);
void save_xyz(char *filename);

void free_memory(void);

void get_positions(void);

void reset_iteration(void);
void update_neighblist(void);

void force_lj(char *types, float *ljparam);
void force_coulomb(float cf);

void integrate_leapfrog(void);

void thermostat_nh(float temp0, float thermostatmass);
void reset_momentum(int resetfreq);

void get_pressure(double *press);
void get_energies(double *energies);
