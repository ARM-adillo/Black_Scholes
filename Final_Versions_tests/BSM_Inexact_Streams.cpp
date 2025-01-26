#include <iostream>
#include <cmath> // Pour std::erf et std::sqrt
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>   // For setting precision
#include <omp.h>
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

// Extract Stream
inline void 
gaussian_box_muller(VSLStreamStatePtr stream, double *buffer, ui64 n)
{
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n, buffer, 0, 1);
}

//
inline double 
fastexp(double x, int n = 4) 
{
	double sum = 1.0;
	double term = 1.0;
	for (int i = 1; i < n; i++) 
	{
	
		term *= x / i;
		sum += term;
	}
 	return sum;
}

//
double 
black_scholes_monte_carlo_t2(
        VSLStreamStatePtr streams, const ui64 S0, 
        const ui64 K, const double drift, const double diffusion, 
        const double res, const ui64 num_simulations)
{
    double sum_payoffs = 0.0;
    ui64 n_threads  = omp_get_num_threads();

    ui64 elt_per_thread = (num_simulations / n_threads);
    const ui64 remainder  = (num_simulations % elt_per_thread);

    std::vector<double> Z_v(num_simulations);
    gaussian_box_muller(streams[tid], Z_V.data(), num_simulations); 


    ui64 n = num_simulations - num_simulations % 8;
    //#pragma omp parallel
    //{
        double Z = 0.0;  double ST = 0.0;
        double Z1 = 0.0; double ST1 = 0.0;
        double Z2 = 0.0; double ST2 = 0.0;
        double Z3 = 0.0; double ST3 = 0.0;
        double Z4 = 0.0; double ST4 = 0.0;
        double Z5 = 0.0; double ST5 = 0.0;
        double Z6 = 0.0; double ST6 = 0.0;
        double Z7 = 0.0; double ST7 = 0.0;
        
        //#pragma omp for simd nowait reduction(+:sum_payoffs) schedule(static)
        for(ui64 i = 0; i < n; i+=8)
        {
            // Load val
            Z 	= Z_v[i] 	* diffusion + drift; 
            Z1 	= Z_v[i+1] 	* diffusion + drift;
            Z2 	= Z_v[i+2] 	* diffusion + drift;
            Z3	= Z_v[i+3]	* diffusion + drift;
            Z4 	= Z_v[i+4] 	* diffusion + drift; 
            Z5 	= Z_v[i+5] 	* diffusion + drift;
            Z6 	= Z_v[i+6] 	* diffusion + drift;
            Z7	= Z_v[i+7]	* diffusion + drift;
            
            // Exp of non const term
            ST 	    = S0 * fastexp(Z);
            ST1 	= S0 * fastexp(Z1);
            ST2 	= S0 * fastexp(Z2);
            ST3 	= S0 * fastexp(Z3);
            ST4 	= S0 * fastexp(Z4);
            ST5 	= S0 * fastexp(Z5);
            ST6 	= S0 * fastexp(Z6);
            ST7 	= S0 * fastexp(Z7);

            sum_payoffs += std::max(ST - K, 0.0);
            sum_payoffs += std::max(ST1 - K, 0.0);
            sum_payoffs += std::max(ST2 - K, 0.0);
            sum_payoffs += std::max(ST3 - K, 0.0);
            sum_payoffs += std::max(ST4 - K, 0.0);
            sum_payoffs += std::max(ST5 - K, 0.0);
            sum_payoffs += std::max(ST6 - K, 0.0);
            sum_payoffs += std::max(ST7 - K, 0.0);
        }
        
        //#pragma omp for simd nowait reduction(+:sum_payoffs) schedule(static) 
        for(ui64 i = n; i < num_simulations; ++i)
        {
            Z = Z_v[i] * diffusion + drift;
            ST = S0 * fastexp(Z);
            sum_payoffs += std::max(ST - K, 0.0);
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
    
    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;

/******************************************************************************************************************************************/
    // Allocate streams, implem based on
    // https://developer.arm.com/documentation/101004/2410/Open-Random-Number-Generation--OpenRNG--Reference-Guide/Examples/skipahead-c?lang=en
    const int brng = VSL_BRNG_SFMT19937;
    const uint32_t n_threads = omp_get_num_threads();

    std::vector<VSLStreamStatePtr> parallel_streams(n_threads);
    vslNewStream(&parallel_streams[0], VSL_BRNG_SFMT19937, global_seed);
    
    for (int i = 1; i < n_threads; i++) {
        vslCopyStream(&parallel_streams[i], parallel_streams[i - 1]);
        vslSkipAheadStream(parallel_streams[i], num_simulations * num_runs / n_threads);
    }

    std::vector<double> Z(num_simulations, 0.0);
    std::vector<double> Z_swap(num_simulations, 0.0);
	
    ui64 elt_per_thread = (num_simulations / n_threads);
    ui64 remainder      = (num_simulations % elt_per_thread);
	
    #pragma omp parallel for schedule(static)
    for(ui64 i = 0; i < n_threads; ++i)
    {
        // That s not proper C++ we all agree
        auto buffer_start = Z.data() + i * elt_per_thread;
        gaussian_box_muller(parallel_streams[i], buffer_start, elt_per_thread);
    }
    // For the remainder in case num_simulations isnt multiple of n_threads
    gaussian_box_muller(parallel_streams[0], Z.data() + elt_per_thread * n_threads, remainder);
/******************************************************************************************************************************************/

    constexpr double sigma_sq_d = (sigma * sigma) / 2.0;
    constexpr double drift      = (r - q - sigma_sq_d) * T;
    const double diffusion      = (sigma * std::sqrt(T));
    const double res_           = std::exp(-r * T); 
 
    sum=0.0;
    double t1=dml_micros();
   
     #pragma omp parallel for reduction(+:sum) schedule(static) 
    for (ui64 run = 0; run < num_runs; ++run) {
	int tid = omp_get_num_threads();
        sum+= black_scholes_monte_carlo_t2(parallel_streams[tid], S0, K, drift, diffusion, res_, num_simulations);
    }

    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;
    
    // Free rand gen streams
    for (int i = 0; i < n_threads; i++) 
    {
        vslDeleteStream(&parallel_streams[i]);
    }
}
