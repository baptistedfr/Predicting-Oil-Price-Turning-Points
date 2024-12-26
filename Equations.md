## Log Periodic Power Law (LPPL)

The LPPL model is described as below : 

$$y(t) = A + B(t_c − t)^{\alpha} + C(t_c − t)^{\alpha} cos[\omega ln(t_c − t) + \phi] \ \ \ \ (1)$$

where $y(t)$ is the oil price at time $t$, $\alpha$ is the exponential growth, $\omega$ is the control of the amplitude of the oscillations and A, B, C and $\phi$ are the parameters with no structural information.

Note that in Eq. (1), $t_c$ is a critical time or a turning point, to be predicated.

One main feature captured by Eq. (1) is the dampened yet accelerated oscillation in oil price. That is, when $t$
approaches $t_c$, the oscillations occur more frequently with a decreasing amplitude. In other words, $(t_c − t)^\alpha$ is the power law term, which describes the faster than exponential change of the prices owing to positive feedback mechanisms and $(_c − t)^\alpha cos[\omega ln(tc −t)+\phi]$ is the periodic term, which indicates a correction to the power law term and has the symmetry of discrete scale invariance. 

The most probable time of the turning point is when $t = t_c$, for $t ≥ t_c$.

In the above LPPL model, there are seven parameters that need to be optimized, including four nonlinear parameters, $t_c$, $\omega$, $\phi$ and $\alpha$; and three linear parameters, A, B and C. 

For simplicity, the linear parameters can be directly derived from the given nonlinear parameters by using least square method.

$$f_j = (t_c - t_j)^\alpha \ \ \ \ (2)$$

$$g_j = (t_c - t_j)^\alpha \cos[\omega ln(t_c - t_j) + \phi] \ \ \ \ (3)$$

where $t_j$ ($j = 1, 2, ... , J$) with $j$ as a time unit and $J$ is the total time units in an interval, the linear parameters can be calculated using the following equations:

$$
\begin{pmatrix}
A \\
B \\
C
\end{pmatrix}
=
[(V^T_{3 \times J} \dot V_{J \times 3})^{-1} (V^T_{3 \times J} \dot V_{J \times 1})]_{3 \times 1}
\ \ \ \ (4)
$$

where 
$
V_{J \times 3} = 
\begin{pmatrix}
 1 & f_1 & g_1 \\ 
 1 & f_2 & g_2 \\ 
 . & . & . \\
 . & . & . \\
 . & . & . \\
 1 & f_J & g_J 
\end{pmatrix}_{J \times 3}
$
, which is a matrix with $J$ rows and 3 columns.


$
Y_{J \times 1} = 
\begin{pmatrix}
y(1) \\ 
y(2) \\ 
 . \\
 . \\
 . \\
y(J)
\end{pmatrix}_{J \times 1}
$
, which is a column vector with $J$ rows.

$$
\begin{pmatrix}
A \\
B \\
C
\end{pmatrix}
= 
\begin{pmatrix}
J & \sum_{j=1}^{J} f_j & \sum_{j=1}^{J} g_j \\
\sum_{j=1}^{J} f_j & \sum_{j=1}^{J} f_j^2 & \sum_{j=1}^{J} f_j g_j \\
\sum_{j=1}^{J} g_j & \sum_{j=1}^{J} f_j g_j & \sum_{j=1}^{J} g_j^2
\end{pmatrix}^{-1}
\begin{pmatrix}
\sum_{j=1}^{J} y(j) \\
\sum_{j=1}^{J} y(j) f_j \\
\sum_{j=1}^{J} y(j) g_j
\end{pmatrix}
\ \ \ \ (5)$$

This approach is proven to be very stable and able to yield good estimation of the linear parameters A, B and C.

linear parameters, $t_c$, $\omega$, $\phi$ and $\alpha$ proves to be more challenging. In fact, it can be proven that searching for the optimal values of the four nonlinear parameters in the LPPL model is an NP-hard problem.

For this purpose, the multi-population genetic algorithm (MPGA) is employed to search for the optimal parameter values in the LPPL model. The MPGA is one of the most popular heuristic algorithms with the advantages of improving convergence rates and maintaining relatively low mean-square-errors.

So the framework of LPPL Model can be briefly summarized as follows:

**Step 1 :** 

A sample interval is selected for the prediction of a turning point in the future time horizon. The interval is at least four-year long after the previous major turning point in history.

**Step 2 :** 

The sample interval is further divided into over 100 subintervals to avoid the bias of specific sample interval and the impact of selecting a sample interval on the forecast result.

**Step 3 :** 

For each subinterval, the MPGA (multi-population genetic algorithm) is employed to optimize the parameters in the LPPL model. The optimized LPPL model is then used to predict a date in the future when a turning point will occur.

**Step 4 :** 

The Lomb periodogram analysis is conducted to statistically test the predicted turning points obtained by the LPPL models for all subintervals.

**Step 5 :** 

The turning points that are statistically validated by the Lomb periodogram analysis are considered as predicted turning points by the LPPL model

---

## Multi-Population Genetic Algorithm (MPGA)

the MPGA works on multiple populations with the objective to evaluate each subinterval. After the initial populations are produced, if the optimization criteria are not met, new populations are created and the search starts again. 

The first step of the MPGA is to generate multiple populations. Inspired by the biology concepts of crossover and mutation, each population in the MPGA can be muted into hundreds of chromosomes and each chromosome represents a feasible solution for the four nonlinear parameters in the LPPL model. Based on Eq. (1), the MPGA measures the fitness value of each of chromosomes (i.e., the four nonlinear parameters) generated from all the populations by computing the residual sum of squares (RSS) between the historical oil price at time t or y(t) and the results from the LPPL models:

$$
RSS_{m,n} = \sum_{t=1}^{J} \left( y(t) - A - B (t^{m,n}_c - t)^{\alpha^{m,n}} - C (t^{m,n}_c - t)^{\alpha^{m,n}} \cos \left[\omega^{m,n} \ln (t^{m,n}_c - t) + \phi^{m,n} \right] \right)^2
$$

where $RSS_{m,n}$ represents the fitness value (RSS) of the $n$-th chromosome in the $m$-th population; $t^{m,n}_c$, $\omega^{m,n}$, $\phi^{m,n}$ and $\alpha^{m,n}$ correspond to the $n$-th chromosome in the $m$-th population.

For each generation, the minimum fitness value and its corresponding chromosome in each population ($\substack{\min \\ n} (RSS_{m,n})$ for each $m$), and the minimum fitness value in all populations ($\substack{\min \\ m,n} (RSS_{m,n})$) and its corresponding chromosome are recorded.

A new population will be generated by selection, crossover, mutation and re-inserting. Next, the chromosome whose fitness value is the smallest among the $m$-th population ($\substack{\min \\ n} (RSS_{m,n})$ for each $m$) will substitute the chromosome whose fitness value is the largest among the $m + 1$-th population ($\substack{\max \\ n} (RSS_{m,n})$ for each $m + 1$). 

This process is generally referred to as the immigration operation that combines individual populations into a unified entity. After the immigration operation, if the minimum fitness value in a new population ($\substack{\min \\ n} (RSS_{m,n})$) is less than the corresponding record (i.e., the minimum fitness value and its corresponding chromosome in this population), the records are updated; otherwise, the original records in this population remain. 

The record of the minimum fitness value of all populations ($\substack{\min \\ m,n} (RSS_{m,n})$) and the corresponding chromosome are processed in the same way. Finally, if the minimum fitness value of all populations does not change for a given number of consecutive iterations (set to 50 in our computation), or the total number of iterations reaches a given upper bound (set to 500 in our computation), the algorithm terminates and the latest minimum fitness value of all populations and its corresponding chromosome ($t_c$, $\omega$, $\phi$ and $\alpha$) are considered as the outputs of the MPGA. After obtaining the four nonlinear parameters optimized by the MPGA, the three linear parameters can be subsequently derived from Eq. (5). The corresponding LPPL model is thus established. 

---

## Validation of turning point prediction using the Lomb periodogram analysis

In order to validate the turning points predicted by the LPPL models, it is necessary to determine whether the frequency ($\frac{\omega}{2\pi}$) obtained from the MPGA (note that $\omega$ is one of the optimal four nonlinear parameters) and the frequency of $y(t) − A − B(t_c − t)^\alpha$ are consistent. A test method called the Lomb periodogram analysis is adopted to detect periodic oscillations for $y(t) − A − B(t_c − t)^\alpha$ and calculate its frequency.

There is many existing methods but the Lomb periodogram analysis method is not only able to objectively evaluate the critical time tc or the turning point, but is also suitable for non-uniform time series.

The validation process starts with pre-setting the frequency series ($freq_i$) ($i = 1, 2, ... , M$) with $M$ as the length of the pre-given frequency series. For a given frequency $f$, the power spectral density $P(f)$ can be computed by the Lomb periodogram analysis as below:

$$
P(f) = \frac{1}{2\sigma^2} \left\{
\frac{\left( \sum_{j=1}^{J} (x_j - \bar{x}) \cos \left( 2\pi f (t_j - \tau) \right) \right)^2}{\sum_{j=1}^{J} \cos^2 \left( 2\pi f (t_j - \tau) \right)}
+
\frac{\left( \sum_{j=1}^{J} (x_j - \bar{x}) \sin \left( 2\pi f (t_j - \tau) \right) \right)^2}{\sum_{j=1}^{J} \sin^2 \left( 2\pi f (t_j - \tau) \right)}
\right\}
$$

where $$x_j = y_j - A - B(t_c - t_j)^\alpha$$ 

at times $t_j$ ($j = 1, 2, ..., J$); 

$$\bar{x} = \frac{1}{J} \sum_{j=1}^{J} x_j$$ 

and 

$$\sigma^2 = \frac{1}{J-1} \sum_{j=1}^{J}(x_j - \bar{x})^2$$ 

are respectively the mean and the variance of $x_j$. The time offset, $\tau$ is calculated by 

$$\tau = \frac{1}{4\pi f} \arctan \frac{\sum_{j=1}^{J}\sin(4\pi f t_j)}{\sum_{j=1}^{J}\cos(4\pi f t_j)}$$

Invalid values are then removed from the resulting $P(freq_i)$ series ($i = 1, 2, ... , M$). These invalid values include: 1) $P(f_{mpf})$ corresponding to the most probable frequency $(f_{mpf})$, which is caused by the random series, and inversely proportional to the length of the given frequency series ($J$), $f_{mpf} ≈ 1.5/J$; 2) the $P(freq_i)$ which is smaller than
the critical value that is calculated by $z = −ln(1 − (1 − p)^{1/M})$, at
the given statistical significance level of p. If there are no valid values
in the $P(freq_i)$ series, the Lomb periodogram rejects the conclusion.

In other words, the turning points predicted by the LPPL model are
not statistically valid. Otherwise the frequency corresponding to the
maximum valid values in the $P(freq_i)$ series is the result of the Lomb
periodogram test.

The Lomb periodogram analysis can be briefly summarized as
follows. First, an LPPL model corresponding to each subinterval is
obtained. 

The Lomb periodogram analysis then computes the frequency value based on the periodic oscillations of the LPPL model.
If the frequency value is close to the frequency $(\omega/2\pi)$ optimized
by the MPGA, the Lomb periodogram analysis concludes that the
prediction by the LPPL model is effective. Otherwise, the predicted
turning points are invalid and are thus deleted. 

Eventually only the
turning points predicated by the LPPL models that pass the Lomb
periodogram test are recorded.

---

### **Algorithm 1.** Pseudo-code of the MPGA to optimize nonlinear parameters.

1: Read historical data;

2: Set the start time and end time of the sample denoted as $time_{start}$ and $time_{end}$ respectively;

3: Predetermine the value ranges of the four nonlinear parameters, $t_c$, $\omega$, $\phi$ and $\alpha$ ($t_c$ is the day after sample to the future 10 years; $\omega$ is between 0 and 40; $\phi$ is between $0$ and $2\pi$; $\alpha$ is between $0.1$ and $0.9$);

4: Predetermine the number of all populations, and let it equal $10$;

5: Predetermine population size, and let it equal $100$;

6: Predetermine the up bound of the total loop denoted as $MaxGen$, and let it equal 500;

7: Predetermine the up bound that the minimum fitness value of all population does not change, denoted as $StopGen$, and let it equal $50$;

8: Predetermine the selection probability of each population, and let it equal 0.9;

9: Selecting the subintervals from the sample is as follows:

10: Predetermine moving step of start time of subintervals, denoted as $delta$, and let it equal the larger of $(time_{end}-time_{start}) \times 0.75 / (three weeks)$ and three weeks;

11: **for** $subinterval_{end} = time_{end}$ : - one week : $time_{end}-$ six weeks **do**

12: - **for** $subinterval_{start} = time_{start} : delta : time_{end} - (time_{end} - time_{start})/4$ **do**

13: - - Get the subinterval $[subinterval_{start} \ \ subinterval_{end}]$ from the sample $[time_{start} \ \ time_{end}]$, where $subinterval_{start}$ is the start time of this subinterval; $subinterval_{end}$ is the end time of this subinterval;

14: - - Randomly generate the crossover probability of each population between 0.001 and 0.05;

15: - - Randomly generate the mutation probability of each population between 0.001 and 0.05;

16: - - **for** each population **do**

17: - - - Initialize one population, which includes 100 chromosomes;

18: - - - Calculate the fitness value of each chromosome, according to Equation 6;

19: - - - Find the minimum fitness value of current population;

20: - - - Record this minimum fitness value, denoted as $bestObjV$, and its corresponding chromosome;

21: - - **end for**

22: - - Find and record the minimum fitness value of all populations;

23: - - Add two counters, one is used to record the current number of loop, denoted as $gen$, and the other is used to record the number that the minimum fitness value of all populations does not change, denoted as $gen0$. Let $gen$ equal 1, and let $gen0$ equal 0;

24: - - **while** $gen0 < StopGen$ and $gen <= MaxGen$ do

25: - - - **for** each population **do**

26: - - - - Perform the selection, crossover, and mutation and reinserting operation;

27: - - - **end for**

28: - - - Perform immigration operation;

29: - - - Find the minimum fitness value of each population and their corresponding chromosomes in the current loop;

30: - - - Find the minimum fitness value of all populations in the current loop, denoted as $newbestObjV$;

31: - - - **if** $newbestObjV < bestObjV$ **then**

32: - - - - $bestObjV = newbestObjV$;

33: - - - - $gen0 = 0$;

34: - - - **else**

35: - - - - $gen0 = gen0 + 1$

36: - - - **end if**

37: - - - $gen = gen + 1$;

38: - - **end while**

39: - - Save $bestObjV$ and its corresponding chromosome under the current subinterval;

40: - **end for**

41: **end for**
