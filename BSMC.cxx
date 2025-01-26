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
#include <sys/time.h>

// Set desired type for precision here
using real = double;

inline double
dml_micros()
{
        static struct timezone tz;
        static struct timeval  tv;
        gettimeofday(&tv,&tz);
        return((tv.tv_sec*1000000.0)+tv.tv_usec);
}

// Different template specialization needed due to function signatures of the armpl
template <typename T>
inline void
logNorm(std::vector<T> &buffer, uint64_t num_simulations, uint64_t global_seed, const T sigma)
{
    std::cout << "Log norm can only be used using float or double on all real parameters\n";
    exit(1);
}

template <>
inline void
logNorm(std::vector<float> &buffer, uint64_t num_simulations, uint64_t global_seed, const float sigma)
{
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, global_seed);
    vsRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream,
                   num_simulations, buffer.data(), 0.0,
                   sigma, 0.0, 1.0);
        vslDeleteStream(&stream);
}

template <>
inline void
logNorm(std::vector<double> &buffer, uint64_t num_simulations, uint64_t global_seed, const double sigma)
{
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, global_seed);
    vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2, stream,
                   num_simulations, buffer.data(), 0.0,
                   sigma, 0.0, 1.0);
        vslDeleteStream(&stream);
}

// Main Black-Scholes function using log normale ditribution in order to
// only compute a reduction
template <typename T>
auto BSMC(
    const T k,
    const T res,
    const uint64_t num_simulations,
    const uint64_t global_seed,
    const T sigma_sqrt
) -> T {

    auto sum_payoffs = 0.0;

    auto Z_v = std::vector<T>(num_simulations);
    logNorm(Z_v, num_simulations, global_seed, sigma_sqrt);

    for(uint64_t i = 0; i < num_simulations; ++i)
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

    uint64_t num_simulations = std::stoull(argv[1]);
    uint64_t num_runs        = std::stoull(argv[2]);

    // Input parameters
    constexpr uint64_t S0       = 100;   // Initial stock price
    constexpr uint64_t K        = 110;   // Strike price
    constexpr real T            = 1.0;   // Time to maturity (1 year)
    constexpr real r            = 0.06;  // Risk-free interest rate
    constexpr real sigma        = 0.2;   // Volatility
    constexpr real q            = 0.03;  // Dividend yield

    // Generate a random seed at the start of the program using random_device
    std::random_device rd;
    uint64_t global_seed = rd();  // This will be the global seed

    // Pre computation of all constant terms
    constexpr real sigma_sq       = sigma * sigma;
    constexpr real Tr             = T * r;
    constexpr real tmp_rq_sigma   = Tr + (-q - 0.5 * sigma_sq) * T;

    // std::exp and std::sqrt are only constexpr in C++26 unfortunately
    const real tmp_st             = S0 * std::exp(tmp_rq_sigma);
    const real sq_and_sigma       = std::sqrt(T) * sigma;
    const real res_init           = tmp_st * 1.0/(std::exp(r * T));
    const real k                  = K  * (1.0 / tmp_st);

    const real res = res_init * (1.0/num_simulations);
    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] <<  std::endl;

    real sum=0.0;
    double t1=dml_micros();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (uint64_t run = 0; run < num_runs; ++run)
    {
        sum+= BSMC(k, res, num_simulations, global_seed, sq_and_sigma);
    }

    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;

    return 0;
}
