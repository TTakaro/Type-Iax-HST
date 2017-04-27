import os
import glob
import platform
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.integrate as integrate
import multiprocessing as mp
from functools import partial


#################### Sets Up Variables for Pulling From Isochrones ######################

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

ages = np.array(list(set(df.log10_isochrone_age_yr)))
ages.sort()
age_cmd = {}

######################## Encodes functions for the Monte Carlo ########################


""" These two functions set up the IMF sampling. The function invSalpeter is the inverse of the cumulative distribution
    for a Salpeter IMF, or equivalently, the quantile function. This is useful because it allows us to draw masses at
    random from an IMF by feeding in random numbers generated from a uniform distribution into the quantile function.
"""
def SalpeterUnNorm(m):
    return m**-2.35
def invSalpeter(u, lower, upper):
    norm = integrate.quad(SalpeterUnNorm, lower, upper)[0] # To go back to how it was, replace the 6's with .075's
    return (lower**(-1.35) - 1.35 * norm * u)**(-1/1.35)

""" This function generates a mass from the IMF, then determines the associated magnitude from the isochrone. It
    does not use interpolation to determine magnitudes, instead just using the magnitude associated with the mass
    nearest to the input mass. Add interpolation to get more precise magnitudes would improve the precision, but
    would also increase the computation time.
"""
def Random_mass_mag(mass, mag4, mag5, mag6, mag8):
    m = invSalpeter(np.random.random(), 4, np.amax(mass))
        
    # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
    loc=np.array([mag4[np.argmin(np.abs(m - mass))], mag5[np.argmin(np.abs(m - mass))], 
                      mag6[np.argmin(np.abs(m - mass))], mag8[np.argmin(np.abs(m - mass))]])
    scale=np.array([7.8e-12 * 2.4**loc[0], 1.25e-11 * 2.4**loc[1], 1.33e-11 * 2.4**loc[2], 1.45e-11 * 2.4**loc[3]])
    #np.random.seed()
    mags = np.random.normal(loc=loc, scale=scale, size=4)
    return np.array([m, mags[0], mags[1], mags[2], mags[3]])


""" This function generates a set of false stars using the errors in magnitude and distance, assuming normal
    distributions. It then performs the Crappy Chi-squared (CChi) statistical test, in order to compare to real
    stars. Some notes: 1. It weights by physical distance, but then divides out the average weight at the end, in
    order to avoid skewing a comparison with real stars, where the average weight may be different. 2. This function
    is written with bad coding practices, as it uses several variables defined outside the function. For this reason,
    be very careful (just don't) using this function outside of the bounds of this script.
"""
def False_Stars_CChi(reddening, age):
    np.random.seed()
    dist = np.random.normal(loc=21.81e6, scale=1.53e6) # Chooses distance using gaussian with errors from literature
    dist_adjust = 5 * (np.log10(dist) - 1) # Converts distance to a magnitude adjustment
    F435W_ext = 0.283 # extinction in F435W in UGC 12682 from NED
    F555W_ext = 0.219 # extinction in F555W in UGC 12682 from NED
    F625W_ext = 0.174 # extinction in F625W in UGC 12682 from NED
    F814W_ext = 0.120 # extinction in F814W in UGC 12682 from NED

    idx = df.log10_isochrone_age_yr == age
    mass = df[idx].initial_mass
    mag_435 = df[idx].ACS_WFC_F435W + dist_adjust + F435W_ext + 3.610*reddening
    mag_555 = df[idx].ACS_WFC_F555W + dist_adjust + F555W_ext + 2.792*reddening
    mag_625 = df[idx].ACS_WFC_F625W + dist_adjust + F625W_ext + 2.219*reddening
    mag_814 = df[idx].ACS_WFC_F814W + dist_adjust + F814W_ext + 1.526*reddening


    # This array will hold 1. Mass 2. Radial distance 3-6. Magnitudes
    False_stars = np.zeros([24, 6]) # Here, 24 corresponds to the number of real stars considered.

    temp = 0 # This will hold the cumulative difference in magnitdue between the stars and isochrone
    phys_dist_temp = 0 # This will hold the comulative phyical distance between the stars and the SN position
    for x in range(False_stars.shape[0]):
        # Generates stars with randomly drawn mass, then finds corresponding magnitude in each filter
        False_stars[x,1], False_stars[x,2], False_stars[x,3], False_stars[x,4], False_stars[x,5] = Random_mass_mag(
            mass, mag_435, mag_555, mag_625, mag_814)
        # Checks to make sure that the magnitude in each filter is above some limiting magnitude.
        while (False_stars[x,2] > 30) or (False_stars[x,3] > 30) or (False_stars[x,4] > 30) or (False_stars[x,5] > 30):
            False_stars[x,1], False_stars[x,2], False_stars[x,3], False_stars[x,4], False_stars[x,5] = Random_mass_mag(
                mass, mag_435, mag_555, mag_625, mag_814)
    
        # Samples radial distribution to get radial distance from SN
        sigma = .92 * 10**age * 3.15e7 * (360 * 60 * 60)/(2 * np.pi) * 1/(21.81e6 * 3.086e13 * .05)
        # Adds in inherent spread in star position at formation with the 100 * rand.rand()
        False_stars[x,1] = abs(np.random.normal(loc=0, scale=sigma)) + 100 * np.random.random()
    
        # Now, determine Crappy Chi-squared fit
        phys_dist_weight = 2 * 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(- False_stars[x,1]**2/(2 * sigma**2)) # Gaussian in radius
        phys_dist_temp += phys_dist_weight # Will be used to compute average of the weights
        # Adds the distance for each data point in quadrature.
        temp = temp + (phys_dist_weight * np.amin(np.sqrt((False_stars[x,2] - mag_435)**2
          + (False_stars[x,3] - mag_555)**2 + (False_stars[x,4] - mag_625)**2 + (False_stars[x,5] - mag_814)**2)))**2
    phys_dist_temp /= False_stars.shape[0]
    #output.put(np.sqrt(temp)/phys_dist_temp)
    return np.sqrt(temp)/phys_dist_temp


######################## Runs the Monte Carlo ########################

ages = ages[(ages > 7.68) & (ages < 7.72)] #ages[(ages >= 6.5) & (ages <= 8.5)]

df = isochrones[-0.50] # Sets metallicity. Eventually this will be varied over.
Gal_ext = 0 # Sets extinction. Eventually this will be varied over

CChi_false = np.zeros([2,1,1000]) # First dimension is age, CChi; Second is varying age; Third is MC runs

# Generates false stars and applies a CChi test 1000 times to get a distribution of values
for i, age in enumerate(ages):
    CChi_false[0,0,:] = age
    func = partial(False_Stars_CChi, 0)
    if __name__ == '__main__':
        pool = mp.Pool(os.cpu_count())
        results = pool.map(func, age * np.ones(1000))
    CChi_false[1,0,:] = list(results)#list(map(func, age * np.ones(100)))
    #for i in range(CChi_false.shape[2]):
    #    CChi_false[1,0,i] = False_Stars_CChi(age, 0)
outfile = "CChi_false_age7.7"
np.save(outfile, CChi_false)

"""
# Define an output queue
output = mp.Queue()
# Setup a list of processes to run
for i, age in enumerate(ages):
    CChi_false[0,0,:] = age

    processes = [mp.Process(target=False_Stars_CChi, args=(age,0,output)) for x in range(4)]

for p in processes:
    p.start()
for p in processes:
    p.join()
CChi_false = [output.get() for p in processes]
"""
