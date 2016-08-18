import NMRfit as nmrft
import os
import matplotlib.pyplot as plt

#Powell alone works
#inDir = "./Data/blindedData/dc_4a_cdcl3_kilimanjaro_25c_1d_1H_1_043016.fid"

#TNC adds value
#inDir = "./Data/blindedData/dc_4e_cdcl3_kilimanjaro_25c_1d_1H_1_050116.fid"

#Even TNC can't get them all
inDir = "./Data/blindedData/dc_4h_cdcl3_kilimanjaro_25c_1d_1H_2_050616.fid"

w, u, v, theta0 = nmrft.varian_process(os.path.join(inDir, 'fid'), os.path.join(inDir, 'procpar'))

# create data object
data = nmrft.Data(w, u, v, theta0)

# bound the data
data.select_bounds(low=3.2, high=3.6)

# interactively select peaks and satellites
# p = data.select_peaks(2)
# s = data.select_satellites(4)

# weights across ROIs
# weights = [['all', 1.0],
#            ['all', 1.0],
#            [s[0].bounds, 100 * p[0].height / s[0].height],
#            [s[1].bounds, 100 * p[0].height / s[1].height],
#            [s[2].bounds, 100 * p[0].height / s[2].height],
#            [s[3].bounds, 100 * p[0].height / s[3].height]]

# initial conditions of the form [theta, r, yOff, sigma_n, mu_n, a_n,...]
# x0 = [data.theta, 0.1, 0,                                  # shared params
#       p[0].width / 10, p[0].loc, p[0].area,       # p1 init
#       p[1].width / 10, p[1].loc, p[1].area,       # p2 init
#       p[1].width / 10, s[0].loc, s[0].area,       # s1 init
#       p[1].width / 10, s[1].loc, s[1].area,       # s2 init
#       p[1].width / 10, s[2].loc, s[2].area,       # s3 init
#       p[1].width / 10, s[3].loc, s[3].area]       # s4 init


#x0 = [theta0, 0.1, 0,
x0 = [theta0, 1.0, 0,
      0.0027432621761354081, 3.4017370050801361, -0.013512294104020406,
      0.0033193472331238816, 3.434001710684198, -0.014852547123152789,
      0.0033193472331238816, 3.265834154202425, -3.4576883447044533e-05,
      0.0033193472331238816, 3.2985877189823043, -2.3302906139299132e-05,
      0.0033193472331238816, 3.5337289825512954, -0.00011102965515766645,
      0.0033193472331238816, 3.5664825473311765, -0.00010143601447693207]

# fit data
fit = nmrft.FitUtility(data, x0, weights=None, fitIm=False, method='Powell', options=None)

# generate result
res = fit.generate_result(scale=1)


def lblpars(i):
    if (i==0):
        print ('Global Parameters:') 
    elif (i==3):
        print ('Major Peak Parameters:') 
    elif (i==9):
        print ('Minor Peak Parameters:') 

print()
print ('SEED PARAMETER VALUES:')
print()

for i in range(0, len(x0), 3):
    lblpars(i)
    print(x0[i:i + 3])

print()
print ('CONVERGED PARAMETER VALUES:')
print()

for i in range(0, len(res.params), 3):
    lblpars(i)
    print(res.params[i:i + 3])

print ()
print ("FINAL ERROR:  ",res.error)

plt.figure(1)
plt.plot(data.w, data.v)
plt.plot(res.w, res.v)
#plt.show()

plt.figure(2)
plt.plot(data.w, data.I)
plt.plot(res.w, res.I)
plt.show()

print()
print ('Moving onto TNC fit:')
print()

# Now we will pass global results onto TNC
x0[:3]=res.params[:3]

# fit data
fit = nmrft.FitUtility(data, x0, weights=None, fitIm=False, method='TNC', options=None)

# generate result
res = fit.generate_result(scale=1)

print()
print ('SEED PARAMETER VALUES:')
print()

for i in range(0, len(x0), 3):
    lblpars(i)
    print(x0[i:i + 3])

print()
print ('CONVERGED PARAMETER VALUES:')
print()

for i in range(0, len(res.params), 3):
    lblpars(i)
    print(res.params[i:i + 3])

print ()
print ("FINAL ERROR:  ",res.error)

plt.figure(1)
plt.plot(data.w, data.v)
plt.plot(res.w, res.v)
#plt.show()

plt.figure(2)
plt.plot(data.w, data.I)
plt.plot(res.w, res.I)
plt.show()
