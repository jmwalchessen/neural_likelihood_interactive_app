# Interactive App for Neural Likelihood
This is an interactive app (run on local computer) to visualize neural likelihood surfaces from the paper:

J. Walchessen, A. Lenzi, and M. Kuusela. Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. Preprint arXiv:2305.04634 [stat.ME], 2023. [arxiv preprint](https://arxiv.org/abs/2305.04634)

Contact Julia Walchessen at jwalches@andrew.cmu.edu with any questions.

## Structure

This code will generate a webpage (running via local host) which you can then use to generate simulated data from spatial processes and their corresponding neural likelihood surfaces. To run webpage, use command: **python main.py**. On the webpage, select the tab *Gaussian Process*. Enter in the variables for seed value and the parameters
of a Gaussian process with exponential kerne (length scale and variance). Note that length scale and variance should
be numbers between 0 and 2 and have at most 2 decimal places. Press enter and after approximately (10 seconds to a few minutes depending on your laptop capibilities), a visualization of a realization of a Gaussian Process on a
25 by 25 grid will appear and the corresponding exact, uncalibrated neural, and calibrated neural likelihood surfaces. So far, we do not have code for Brown--Resnick processes.

**The package requirements to run this code are in requirements.txt**


