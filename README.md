The nmrfit module reads the output from an NMR spectroscopy experiment and, through a number of intuitive API calls, produces a least-squares fit of the data using Voigt-body approximations such that relative isotope abundance can be easily calculated.  

To read input data, nmrfit relies on the [nmrglue](https://www.nmrglue.com/ "nmrglue homepage") package.  Real and imaginary spectrum components are stored in a custom class, allowing data preprocessing operations to naturally extend from the object as methods.  The data is then processed by Python script using the nmrfit API.

```python
# import the module
import NMRfit as nmrfit

# read in data
nmrfit.load('fid', 'propcar')
```
In many cases, the signal of interest comprises only a subsection of the captured spectrum.  To restrict the fitting algorithm to only the pertinent part of the signal, the method get_bounds() is used to bound the data with respect to frequency.  The lower and upper bounds may be passed as arguments, or no arguments may be passed to prompt the user to interactively select the bounds by clicking twice on a displayed plot of the data.  To prepare for subsequent steps, nmrglue package is again used to perform an initial, approximate phase correction (initial phase correction is later refined by the fitting process).

```python
# bound the data interactively
data.select_bounds()

# alternatively, pass the bound
data.select_bounds(low=3.23, high=3.60)

# phase correction
data.shift_phase(method='auto')
```

In order to seed the optimization method, approximate initial conditions must be extracted from the data.  This is achieved by determining the total number of peaks, finding each peak's center, width, and area, and then using this information to construct initial-condition and weight vectors.  The user may perform this manually--clicking twice per peak to define peak bounds, whereafter peak attributes are calculated--or automatically--wherein a peak-detection algorithm is run--through the method select_peaks().  In either case, the plot flag may be enabled to visualize the results of the peak selection process.  Note that in the manual case, a flag specifiying the number of peaks must be passed.

```python
# select peaks automatically
peaks = data.select_peaks(method='auto', plot=True)

# select peaks manually
peaks = data.select_peaks(method='manual', n=6, plot=True
```

The generate_solution_bounds() method is then used to create upper and lower bounds for the fit by least-squares minimization.  Each set of bounds (lower, upper) contains 3 global parameters (phase, theta; Gaussian-Lorentzian ratio, r; and y-offset, dy) and 3 parameters per peak (width;  center, mu; and area, a).  These values are used to construct area-parameterized Voigt-body approximations for each peak [2009 paper].  

```python
# generate the solution bounds
lb, ub = data.generate_solution_bounds()
```

A FitUtility object is initialized with the Data object and solution bounds to perform a fit via minimization.  Each time the optimizer calls the objective function, the target data is phase-shifted by theta, Voigt-body approximations are generated for each peak and summed to create a fit of the entire signal, and a residual is calculated between the fit and the data.

Once the optimizer converges, the FitUtility method generate_result() generates the final fit from the solution vector and is returned as a Result object, which contains attributes that store residual error, the fit parameter vector, and real and imaginary components of the phase-corrected and out-of-phase fits.  The scale flag may be adjusted to upsample the resulting fit by a constant factor.  This is useful when high-resolution output is desired.  Finally, fit summary statistics and plots of the fit can be viewed using the summary() method of the FitUtility object.

```python
# perform the fit
fit = nmrft.FitUtility(data, lb, ub)

# generate results
res = fit.generate_result(scale=1)

# summary
fit.summary()
```
