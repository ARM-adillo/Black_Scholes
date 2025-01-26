#include <iostream>
#include <cmath> // Pour std::erf et std::sqrt
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>   // For setting precision
#include <omp.h>
#include <tuple>
#include <armpl.h>

using ui64 = u_int64_t;

#include <sys/time.h>

inline double
dml_micros()
{
        static struct timezone tz;
        static struct timeval  tv;
        gettimeofday(&tv,&tz);
        return((tv.tv_sec*1000000.0)+tv.tv_usec);
}

static std::vector<unsigned int> seeds; 
inline std::tuple<double, double> customBoxMuller(){ 
	double u1 = 0 ;
	double u2 = 0 ;
	double r = 0 ;

	auto threadId = omp_get_thread_num();
	do{
	    u1 = 2 * (rand_r(&(seeds[threadId])) + 1.0E-5) / (RAND_MAX) - 1;
	    u2 = 2 * (rand_r(&(seeds[threadId])) + 1.0E-5) / (RAND_MAX) - 1;
			    r = u1 * u1 + u2 * u2;
	}while (r >= 1) ;

	double log_r = sqrt(-2 *log(r) / r);

	return {u1 * log_r , u2 * log_r};
}

inline void gaussian_box_muller(std::vector<double> &Z, ui64 num_simulations) 
{
    ui64 n = num_simulations - num_simulations & 1;
   
    #pragma omp parallel for
    for(int i = 0; i < n; i = i + 2)
    {   
        auto tmp = customBoxMuller();
        Z[i]        = std::get<0>(tmp);
        Z[i + 1]    = std::get<1>(tmp); 
    }

    if(n != num_simulations)
    {
	auto tmp = customBoxMuller();
	Z[num_simulations] = std::get<0>(tmp);
    }
}

//
double black_scholes_monte_carlo(std::vector<double> &results, const ui64 S0, const ui64 K, const double drift,
        const double diffusion, const double res, const ui64 num_simulations)
{
    double sum_payoffs = 0.0;
    double Z = 0.0;  double ST = 0.0;
    double Z1 = 0.0; double ST1 = 0.0;
    
    std::vector<double> Z_v(num_simulations);
    gaussian_box_muller(Z_v, num_simulations);

    for(ui64 i = 0; i < num_simulations; ++i)
    {
        // Load val
        Z 	= Z_v[i] * diffusion + drift; 
        
        // Exp of non const term
        ST 	= exp(Z);

        // Store back for reduction afterwards
        results[i]   = S0 * ST;
    }
    
    for(ui64 i = 0; i < num_simulations; ++i)
    {
	    sum_payoffs += std::max(results[i] - K, 0.0);
    }

    return res * (sum_payoffs * (1.0/num_simulations));
}

//
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs        = std::stoull(argv[2]);

    // Input parameters
    constexpr ui64 S0      = 100;   // Initial stock price
    constexpr ui64 K       = 110;   // Strike price
    constexpr double T     = 1.0;   // Time to maturity (1 year)
    constexpr double r     = 0.06;  // Risk-free interest rate
    constexpr double sigma = 0.2;   // Volatility
    constexpr double q     = 0.03;  // Dividend yield 

    double sum=0.0;
    // Generate a random seed at the start of the program using random_device
    std::random_device rd;
    unsigned long long global_seed = rd();  // This will be the global seed

	for (int i = 0 ; i < omp_get_num_threads() ; ++i){
		        seeds[i] = UINT32_MAX / (i + 1);
	}

/**********************************************************************************************************************************************/
    std::cout << "=== Simple version ===\n";
    constexpr double sigma_sq_d = (sigma * sigma) / 2.0;
    constexpr double drift      = (r - q - sigma_sq_d) * T;
    const double diffusion  = (sigma * std::sqrt(T));
    const double res_       = std::exp(-r * T); 

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;
    std::vector<double> results(num_simulations, 0.0);

    sum=0.0;
    double t1=dml_micros();
    for (ui64 run = 0; run < num_runs; ++run) {
        sum+= black_scholes_monte_carlo(results, S0, K, drift, diffusion, res_, num_simulations);
    }
    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;
}
