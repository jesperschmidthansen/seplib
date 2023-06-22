/******
 * 
 * ***************/


void load_xyz(const char xyzfile[]);
void free_memory(void);
void get_positions(void);
void reset_iteration(void);
void update_neighblist(void);
void force_lj(const char *types, float *ljparam);
