/* 

    Monte Carlo Hackathon created by Hafsa Demnati and Patrick Demichel @ Viridien 2024
    The code compute a Call Option with a Monte Carlo method and compare the result with the analytical equation of Black-Scholes Merton : more details in the documentation

    Compilation : g++ -O BSM.cxx -o BSM

    Exemple of run: ./BSM #simulations #runs

./BSM 100 1000000
Global initial seed: 21852687      argv[1]= 100     argv[2]= 1000000
 value= 5.136359 in 10.191287 seconds

./BSM 100 1000000
Global initial seed: 4208275479      argv[1]= 100     argv[2]= 1000000
 value= 5.138515 in 10.223189 seconds
 
   We want the performance and value for largest # of simulations as it will define a more precise pricing
   If you run multiple runs you will see that the value fluctuate as expected
   The large number of runs will generate a more precise value then you will converge but it require a large computation

   give values for ./BSM 100000 1000000        
               for ./BSM 1000000 1000000
               for ./BSM 10000000 1000000
               for ./BSM 100000000 1000000
  
   We give points for best performance for each group of runs 

   You need to tune and parallelize the code to run for large # of simulations

*/

#include <iostream>
#include <cmath> // Pour std::erf et std::sqrt
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>   // For setting precision
#include <omp.h>
#include <tuple>

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

// Function to generate Gaussian noise using Box-Muller transform
double gaussian_box_muller() {
    static std::mt19937 generator(std::random_device{}());
    static std::normal_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

// Function to calculate the Black-Scholes call option price using Monte Carlo method
double black_scholes_monte_carlo(ui64 S0, ui64 K, double T, double r, double sigma, double q, ui64 num_simulations) {
    double sum_payoffs = 0.0;
    for (ui64 i = 0; i < num_simulations; ++i) {
        double Z = gaussian_box_muller();
        double ST = S0 * exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        double payoff = std::max(ST - K, 0.0);
        sum_payoffs += payoff;
    }
    return exp(-r * T) * (sum_payoffs / num_simulations);
}

// Function to generate Gaussian noise using Box-Muller transform
inline void gaussian_box_muller_2(std::vector<double> &Z, ui64 num_simulations) {
    static std::mt19937 generator(std::random_device{}());
    static std::normal_distribution<double> distribution(0.0, 1.0);

    #pragma omp parallel for private(generator, distribution)
    for(ui64 i = 0; i < num_simulations; i++)
    {
        Z[i] = distribution(generator);
    }
}

//
inline std::pair<double, double> customBoxMuller(unsigned int * state){
    double u1 = 0 ;
    double u2 = 0 ;
    double r = 0 ;
    do{
        u1 = 2 * (rand_r(state) + 1.0E-5) / (RAND_MAX) - 1;
        // *state = *state + 1 ;
        u2 = 2 * (rand_r(state) + 1.0E-5) / (RAND_MAX) - 1;
        r = u1 * u1 + u2 * u2;
    }while (r >= 1) ;

    double log_r = sqrt(-2 *log(r) / r);

    return {u1 * log_r , u2 * log_r};
}

static std::vector<unsigned int> seeds(omp_get_num_threads()) ;
inline void gaussian_box_muller_3(std::vector<double> &Z, ui64 num_simulations)
{
    #pragma omp parallel for
    for(int i = 0; i < num_simulations; i = i + 2)
    {
        // unsigned int state = omp_get_thread_num() ^ time(NULL);
        auto [z1, z2] = customBoxMuller(&(seeds[omp_get_thread_num()]));
        Z[i]        = z1;
        Z[i + 1]    = z2;
    }
}

//
inline double expTaylor(double x, int n = 16) {
    double sum = 1.0;
    double term = 1.0;
    for (int i = 1; i < n; i++) {
        term *= x / i;
        sum += term;
    }

    return sum;
}

//
double black_scholes_monte_carlo_t2(std::vector<double> &Z_v, std::vector<double> &results, const ui64 S0, const ui64 K, const double drift,
        const double diffusion, const double res, const ui64 num_simulations)
{
    double sum_payoffs = 0.0;
    double ST = 0.0, Z = 0.0;

    #pragma omp parallel for simd private(ST, Z)
    for(ui64 i = 0; i < num_simulations; ++i)
    {
        // Load val
        Z = Z_v[i] * diffusion + drift; //gaussian_box_muller(); // Z_v[i];
        // Exp of non const term
        ST = expTaylor(Z);
        // Store back for reduction afterwards
        results[i] = S0 * ST;
    }

    #pragma omp parallel for reduction(+:sum_payoffs)
    for(ui64 i = 0; i < num_simulations; ++i)
    {
        sum_payoffs += std::max(results[i] - K, 0.0); //Z_v[i] * (Z_v[i] > 0.0);
    }

    return res * (sum_payoffs * (1.0/num_simulations));
}

//
double black_scholes_monte_carlo_bigbrain(const std::vector<double> &Z_v, std::vector<double> &results, const double sq_and_sigma, 
        const double k, const double res, const ui64 num_simulations)
{
    double sum_payoffs = 0.0;
    double ST = 0.0;

    #pragma omp parallel for simd private(ST)  
    for(ui64 i = 0; i < num_simulations; i++)
    {
        ST          = Z_v[i] * sq_and_sigma;
        results[i]  = expTaylor(ST);
    }

    #pragma omp parallel for reduction(+:sum_payoffs)
    for (size_t i = 0; i < num_simulations; i++)
    {
        sum_payoffs += std::max(results[i] - k, 0.0);
    }

    return res * sum_payoffs;
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
    
    for(int i = 0 ; i < omp_get_num_threads() ; i++){
        seeds[i] = INT32_MAX / omp_get_num_threads() * i;
    }


   /* std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;

    double sum1=0.0;
    double t1=dml_micros();
    for (ui64 run = 0; run < num_runs; ++run) {
        sum1+= black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations);
    }
    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum1/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;*/

/**********************************************************************************************************************************************/
    std::cout << "=== Simple version ===\n";
    constexpr double sigma_sq_d = (sigma * sigma) / 2.0;
    constexpr double drift      = (r - q - sigma_sq_d) * T;
    constexpr double diffusion  = (sigma * std::sqrt(T));
    constexpr double res_       = std::exp(-r * T); 

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;
    std::vector<double> Z_v1(num_simulations);
    std::vector<double> results(num_simulations);
    gaussian_box_muller_3(Z_v1, num_simulations);

    sum=0.0;
    double t3=dml_micros();
    for (ui64 run = 0; run < num_runs; ++run) {
        sum+= black_scholes_monte_carlo_t2(Z_v1, results, S0, K, drift, diffusion, res_, num_simulations);
    }
    double t4=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t4-t3)/1000000.0 << " seconds" << std::endl;
/**********************************************************************************************************************************************/
    std::cout << "=== Big brain v1.0 du GOAT : === \n";
   
    constexpr double sigma_sq       = sigma * sigma;
    constexpr double Tr             = T * r;
    constexpr double tmp_rq_sigma   = Tr + (-q - 0.5 * sigma_sq) * T;
    constexpr double tmp_st         = S0 * std::exp(tmp_rq_sigma);
    constexpr double sq_and_sigma   = sqrt(T) * sigma;
    constexpr double res_init       = tmp_st * 1.0/(std::exp(r * T));
    constexpr double k              = K  * (1.0 / tmp_st);

    const double res = res_init * (1.0/num_simulations);
    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;
    
    std::vector<double> Z_v(num_simulations);
    std::vector<double> results_bb(num_simulations);
    gaussian_box_muller_3(Z_v, num_simulations);
    
    sum=0.0;
    double t7=dml_micros();
    for (ui64 run = 0; run < num_runs; ++run) {
        sum+= black_scholes_monte_carlo_bigbrain(Z_v,results_bb, sq_and_sigma, k, res, num_simulations);
    }
    double t8=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t8-t7)/1000000.0 << " seconds" << std::endl;
 
/**********************************************************************************************************************************************/
    std::cout << "=== Big brain v1.0 du GOAT : === \n";
    
    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;
    
    std::vector<double> Z_v2(num_simulations);
    std::vector<double> results_bb2(num_simulations);
    gaussian_box_muller_2(Z_v2, num_simulations);
    
    sum=0.0;
    double t7b=dml_micros();
    for (ui64 run = 0; run < num_runs; ++run) {
        sum+= black_scholes_monte_carlo_bigbrain(Z_v2,results_bb2, sq_and_sigma, k, res, num_simulations);
    }
    double t8b=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t8b-t7b)/1000000.0 << " seconds" << std::endl;

    return 0;
}

