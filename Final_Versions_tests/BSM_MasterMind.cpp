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

inline void logNorm(std::vector<double> &Z_v, ui64 num_simulations, ui64 global_seed, const double sigma)
{
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_SFMT19937, global_seed);
    vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream,
                   num_simulations, Z_v.data(), 0.0,
                   sigma, 0.0, 1.0);
	vslDeleteStream(&stream);
}

double black_scholes_monte_carlo_mastermind(const double k, const double res, const ui64 num_simulations, const unsigned long long global_seed, const double sigma_sqrt)
{
    double sum_payoffs = 0.0;

    std::vector<double> Z_v = std::vector<double>(num_simulations);
    logNorm(Z_v, num_simulations, global_seed, sigma_sqrt);

    for(ui64 i = 0; i < num_simulations; ++i)
    {
	    sum_payoffs += std::max(Z_v[i] - k, 0.0);
    }
    return res * sum_payoffs;
}

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

    // Generate a random seed at the start of the program using random_device
    std::random_device rd;
    unsigned long long global_seed = rd();  // This will be the global seed
					    
    // Pre computation of all constant terms
    constexpr double sigma_sq       = sigma * sigma;
    constexpr double Tr             = T * r;
    constexpr double tmp_rq_sigma   = Tr + (-q - 0.5 * sigma_sq) * T;
    const double tmp_st             = S0 * std::exp(tmp_rq_sigma);
    const double sq_and_sigma       = std::sqrt(T) * sigma;
    const double res_init           = tmp_st * 1.0/(std::exp(r * T));
    const double k                  = K  * (1.0 / tmp_st);

    const double res = res_init * (1.0/num_simulations);
    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;
    
    constexpr double sigma_sq_d = (sigma * sigma) / 2.0;
    constexpr double drift      = (r - q - sigma_sq_d) * T;
    const double diffusion      = (sigma * std::sqrt(T));
    const double res_           = std::exp(-r * T); 
    
    double sum=0.0;
    double t1=dml_micros();
   
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (ui64 run = 0; run < num_runs; ++run) 
    {
        sum+= black_scholes_monte_carlo_mastermind(k, res, num_simulations, global_seed, sq_and_sigma);    
    }

    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;
 
    return 0;
}

