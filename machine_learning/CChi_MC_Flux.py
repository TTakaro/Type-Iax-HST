import os
import sys
import glob
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.integrate as integrate
import multiprocessing as mp
from functools import partial


############################## Command Line Arguments ##############################

if len(sys.argv) > 1: # Checks for any command line arguments
    if str(sys.argv[1]) == '08ha':
        print("Running with SN 2008ha parameters.")
        nstars = 26
        distance = 21.81e6
        distance_error = 1.53e6
        F435W_ext = 0.283 # extinction in F435W in UGC 12682 from NED
        F555W_ext = 0.219 # extinction in F555W in UGC 12682 from NED
        F625W_ext = 0.174 # extinction in F625W in UGC 12682 from NED
        F814W_ext = 0.120 # extinction in F814W in UGC 12682 from NED
        metallicity = -0.50
    if str(sys.argv[1]) == '10ae':
        print("Running with SN 2010ae parameters.")
        nstars = 29
        distance = 11.0873e6
        distance_error = 1.02266e6
        F435W_ext = .509 # extinction in F435W in ESO 162-17 from NED
        F555W_ext = .394 # extinction in F555W in ESO 162-17 from NED
        F625W_ext = .313 # extinction in F625W in ESO 162-17 from NED
        F814W_ext = .215 # extinction in F814W in ESO 162-17 from NED
        metallicity = -0.75
    if str(sys.argv[1]) == '10el':
        print("Running with SN 2010el parameters.")
        nstars = 50 # THIS IS WRONG!
        distance = 5.63e6
        distance_error = 1.09e6
        F435W_ext = 0.033 # extinction in F435W in NGC 1566 from NED
        F555W_ext = 0.025 # extinction in F555W in NGC 1566 from NED
        F625W_ext = .021 # extinction in F625W in NGC 1566 from NED
        F814W_ext = .014 # extinction in F814W in NGC 1566 from NED
        metallicity = 0.50
else: # If no arguments given, uses the arguments for SN 2008ha
    nstars = 26
    distance = 21.81e6
    distance_error = 1.53e6
    F435W_ext = 0.283 # extinction in F435W in UGC 12682 from NED
    F555W_ext = 0.219 # extinction in F555W in UGC 12682 from NED
    F625W_ext = 0.174 # extinction in F625W in UGC 12682 from NED
    F814W_ext = 0.120 # extinction in F814W in UGC 12682 from NED
    metallicity = -0.50


#################### Sets Up Variables for Pulling From Isochrones ######################

# Checks operating system, to adjust filesystem to work on both.
if platform.system() == "Darwin":
    mist_dir = "/Users/tktakaro/Documents/Type-Iax-HST/MIST_v1.0_HST_ACSWF"
if platform.system() == "Windows":
    mist_dir = "C:/Users/Tyler/Documents/9. UCSC/Research/Type-Iax-HST-master/MIST_v1.0_HST_ACSWF"

kwargs = {"names": ["EEP", "log10_isochrone_age_yr", "initial_mass", "log_Teff", "log_g",
                    "log_L", "z_surf", "ACS_WFC_F435W", "ACS_WFC_F475W", "ACS_WFC_F502N",
                    "ACS_WFC_F550M", "ACS_WFC_F555W", "ACS_WFC_F606W", "ACS_WFC_F625W", 
                    "ACS_WFC_F658N", "ACS_WFC_F660N", "ACS_WFC_F775W", "ACS_WFC_F814W",
                    "ACS_WFC_F850LP", "ACS_WFC_F892N", "phase"],
         "delim_whitespace": True, "comment": "#"}
isochrones = {}
for filename in glob.glob(mist_dir + "/*.iso.cmd"):
    filename = filename.replace("\\", "/")
    feh_string = filename.split("/")[-1].split("_")[3] # Pulls metalicity information
    if feh_string[0] == "p":
        feh = float(feh_string[1:]) # feh is [Fe/H]
    elif feh_string[0] == "m":
        feh = -float(feh_string[1:])
    else:
        raise ValueError
    df = pandas.read_csv(filename, **kwargs)
    df['ages'] = 10 ** df.log10_isochrone_age_yr / 1e9
    isochrones[feh] = df # Creates dictionary accessible by entering a particular metalicity


######################## Encodes functions for the Monte Carlo ########################

""" These two functions set up the IMF sampling. The function invSalpeter is the inverse of the cumulative distribution
    for a Salpeter IMF, or equivalently, the quantile function. This is useful because it allows us to draw masses at
    random from an IMF by feeding in random numbers generated from a uniform distribution into the quantile function.
"""
def SalpeterUnNorm(m):
    return m**-2.35
def invSalpeter(u, lower, upper):
    norm = integrate.quad(SalpeterUnNorm, lower, upper)[0]
    return (lower**(-1.35) - 1.35 * norm * u)**(-1/1.35)

""" This function generates a mass from the IMF, then determines the associated magnitude from the isochrone. It
    does not use interpolation to determine magnitudes, instead just using the magnitude associated with the mass
    nearest to the input mass. Add interpolation to get more precise magnitudes would improve the precision, but
    would also increase the computation time.
"""
def Random_mass_mag(mass, mag4, mag5, mag6, mag8):
    m = invSalpeter(np.random.random(), 4, np.amax(mass))
        
    # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass  
    loc=np.array([10**(-.4 * mag4[np.argmin(np.abs(m - mass))]), 10**(-.4 * mag5[np.argmin(np.abs(m - mass))]), 
                      10**(-.4 * mag6[np.argmin(np.abs(m - mass))]), 10**(-.4 * mag8[np.argmin(np.abs(m - mass))])])
    if len(sys.argv) == 1:
        scale=np.array([1.65e-12 + 1.5e-7*np.sqrt(loc[0]), 2.65e-12 + 1.5e-7*np.sqrt(loc[1]), 2.85e-12 + 1.8e-7*np.sqrt(loc[2]),
                        3.3e-12 + 1.65e-7*np.sqrt(loc[3])])
    elif str(sys.argv[1]) == '08ha':
        scale=np.array([1.65e-12 + 1.5e-7*np.sqrt(loc[0]), 2.65e-12 + 1.5e-7*np.sqrt(loc[1]), 2.85e-12 + 1.8e-7*np.sqrt(loc[2]),
                        3.3e-12 + 1.65e-7*np.sqrt(loc[3])])
    elif str(sys.argv[1]) == '10ae':
        scale=np.array([2.2e-12 + 1.9e-7*np.sqrt(loc[0]), 3.2e-12 + 2.3e-7*np.sqrt(loc[1]), 3.9e-12 + 2.1e-7*np.sqrt(loc[2]),
                        4.8e-12 + 2.25e-7*np.sqrt(loc[3])])
    elif str(sys.argv[1]) == '10el': # Fix this
        scale=np.array([2.2e-12 + 1.9e-7*np.sqrt(loc[0]), 3.2e-12 + 2.3e-7*np.sqrt(loc[1]), 3.9e-12 + 2.1e-7*np.sqrt(loc[2]),
                        4.8e-12 + 2.25e-7*np.sqrt(loc[3])])
    fluxs = np.random.normal(loc=loc, scale=scale, size=4)
    # Computes signal to noise ratio, to be used in selecting observable stars
    N = np.sqrt(1/(scale[0]**-2 + scale[1]**-2 + scale[2]**-2 + scale[3]**-2))
    SN = N * (fluxs[0]/scale[0]**2 + fluxs[1]/scale[1]**2 + fluxs[2]/scale[2]**2 + fluxs[3]/scale[3]**2)
    return np.array([m, SN, fluxs[0], fluxs[1], fluxs[2], fluxs[3]])


""" This function generates a set of false stars using the errors in magnitude and distance, assuming normal
    distributions. It then performs the Crappy Chi-squared (CChi) statistical test, in order to compare to real
    stars. Some notes: 1. It weights by physical distance, but then divides out the average weight at the end, in
    order to avoid skewing a comparison with real stars, where the average weight may be different. 2. This function
    is written with bad coding practices, as it uses several variables defined outside the function. For this reason,
    be very careful (just don't) using this function outside of the bounds of this script.
"""
def False_Stars_CChi(reddening, age):
    np.random.seed()
    # Chooses distance using gaussian with errors from literature
    dist = np.random.normal(loc=distance, scale=distance_error)
    dist_adjust = 5 * (np.log10(dist) - 1) # Converts distance to a magnitude adjustment

    idx = df.log10_isochrone_age_yr == age
    mass = df[idx].initial_mass
    mag_435 = df[idx].ACS_WFC_F435W + dist_adjust + F435W_ext + 3.610*reddening
    mag_555 = df[idx].ACS_WFC_F555W + dist_adjust + F555W_ext + 2.792*reddening
    mag_625 = df[idx].ACS_WFC_F625W + dist_adjust + F625W_ext + 2.219*reddening
    mag_814 = df[idx].ACS_WFC_F814W + dist_adjust + F814W_ext + 1.526*reddening


    # This array will hold 1. Mass 2. Radial distance 3-6. Magnitudes
    False_stars = np.zeros([nstars, 6])

    temp = 0 # This will hold the cumulative difference in magnitdue between the stars and isochrone
    phys_dist_temp = 0 # This will hold the comulative phyical distance between the stars and the SN position
    for x in range(False_stars.shape[0]):
        # Generates stars with randomly drawn mass, then finds corresponding flux in each filter
        False_stars[x,0], SN, False_stars[x,2], False_stars[x,3], False_stars[x,4], False_stars[x,5] = Random_mass_mag(
            mass, mag_435, mag_555, mag_625, mag_814)
        # Checks to make sure that the S/N ratio is high enough, and there is positive flux in each filter
        while SN < 3:
            False_stars[x,0], SN, False_stars[x,2], False_stars[x,3], False_stars[x,4], False_stars[x,5] = Random_mass_mag(
            mass, mag_435, mag_555, mag_625, mag_814)
    
        # Samples radial distribution to get radial distance from SN
        sigma = .92 * 10**age * 3.15e7 * (360 * 60 * 60)/(2 * np.pi) * 1/(distance * 3.086e13 * .05) #10**age
        # Adds in inherent spread in star position at formation with the .1 * rand.rand()
        False_stars[x,1] = abs(np.random.normal(loc=0, scale=sigma)) + .1 * np.random.random()
    
        # Now, determine Crappy Chi-squared fit
        # Convolves a normal distribution with a flat distribution to get distribution used above to generate radius
        weight_func = np.convolve(1/(np.sqrt(2 * np.pi) * sigma) * np.exp(- np.linspace(-2,2,1000)**2/(2 * sigma**2)),
                                 np.append(np.zeros(488),np.append(np.ones(25),np.zeros(487))))
        # Finds where in the convolved array the generated radius falls
        phys_dist_weight = weight_func[1000 + int(False_stars[x,1]*2.5)]
        phys_dist_temp += phys_dist_weight # Will be used to compute average of the weights

        # Adds the magnitude difference for each data point in quadrature.
        temp += (phys_dist_weight * np.amin(np.sqrt((False_stars[x,2] - 10**(-.4*mag_435))**2
          + (False_stars[x,3] - 10**(-.4*mag_555))**2 + (False_stars[x,4] - 10**(-.4*mag_625))**2
          + (False_stars[x,5] - 10**(-.4*mag_814))**2)))**2
    phys_dist_temp /= False_stars.shape[0]
    return np.sqrt(temp)/phys_dist_temp


######################## Runs the Monte Carlo ########################

df = isochrones[metallicity] # Sets metallicity. Eventually this will be varied over.
ages = np.array(list(set(df.log10_isochrone_age_yr)))
ages.sort()
age_cmd = {}
ages = ages[(ages > 6.49) & (ages < 8.51)] # Sets ages to consider.

Gal_ext = 0 # Sets extinction. Eventually this will be varied over

CChi_false = np.zeros([2,ages.size,5000]) # First dimension is age, CChi; Second is varying age; Third is MC runs
CChi = np.zeros([2,5000])

masses = []
# Generates false stars and applies a CChi test 1000 times to get a distribution of values
for i, age in enumerate(ages):
    CChi_false[0,i,:] = age
    func = partial(False_Stars_CChi, 0)
    if __name__ == '__main__':
        pool = mp.Pool(os.cpu_count())
        print("Working on age={Age}".format(Age=np.round(age,decimals=2)))
        results = pool.map(func, age * np.ones(5000))
        CChi[1,:] = list(results)
        pool.close()
        out = "CChi_false_{Age}".format(Age=np.round(age,decimals=2)) # Saves each age separately
        np.save(out, CChi)

    CChi_false[1,i,:] = CChi[1,:]


outfile = "CChi_false_ages" # Saves all ages together
np.save(outfile, CChi_false)