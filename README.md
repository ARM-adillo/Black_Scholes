# Black Scholes
This repository contains the 2024 Teratec Hackathon work of team ARM-adillo on black scholes equation optimization

# Optimizations steps
-> Constants extraction

-> 4 versions 
        base,big brain, megamind, mastermind 

-> Fast exponential -> Taylor  

-> bottlenecks : random number generation
                 exponential (heavy computation)

-> First idea : 
    -> It s own loop
    -> our own generator
    -> parallel generator
    -> arm optimized parallel generator
    -> Extraction of random generator inside random pool
    
-> Loop unrolling, vectorization, isolation of reductions (aborted) 

-> Monte Carlo : Highly Parallel
    -> open mp

    3 versions : 
        -> multiple parallel for
        -> parallel + multiple for
        -> parallel + multiple for + nowait
    
    -> Final version parallelization of the runs
