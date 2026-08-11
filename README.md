# Staggered_Fermions_HMC
Code for Lattice SU(3) gauge theory with staggered fermions on the lattice sites. The algorithm followed is the hybrid monte carlo update as described in Gattringer with the staggered dirac operator. Running the code you can access the following observables at equilibrium: 

-Avg Plaq
-Action
-Polyakov Loops and analysis of string tension

In addition, one has the option to utilize 4HEX smearing adapted from https://arxiv.org/pdf/1011.1780 when running the simulation. There is also access to helper functions curated with the help of gemini that check the correctness of the HMC algorithm and the analytic link derivatives used 
