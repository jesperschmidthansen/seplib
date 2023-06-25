/******
 * 
 * ***************/


void load_xyz(const char xyzfile[]);
void free_memory(void);
void get_positions(void);
void reset_iteration(void);
void update_neighblist(void);
void force_lj(char *types, float *ljparam);
void integrate_leapfrog(void);
void save_xyz(char *filename);
