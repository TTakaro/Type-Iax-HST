import os
import sys
import glob
import platform
import numpy as np
import pandas
import scipy.integrate as integrate
import multiprocessing as mp
from functools import partial
import time


############################## Command Line Arguments ##############################

if len(sys.argv) > 1: # Checks for any command line arguments
    if str(sys.argv[1]) == '08ha':
        print("Running with SN 2008ha parameters.")
        nstars = 14 # Normally is 15, just testing with new dolphot output
        distance = 21.81e6
        distance_error = 1.53e6
        filters = ["F435W", "F555W", "F625W", "F814W"]
        F435W_ext = 0.283 # extinction in F435W in UGC 12682 from NED
        F555W_ext = 0.219 # extinction in F555W in UGC 12682 from NED
        F625W_ext = 0.174 # extinction in F625W in UGC 12682 from NED
        F814W_ext = 0.120 # extinction in F814W in UGC 12682 from NED
        Zeropoints = np.array([25.0164, 24.9694, 24.9904, 24.7724]) + np.log10(2)
        Exposure_times = 1.65 * np.array([1164, 764, 840, 1240])
        WFC3 = False # Are there WFC3 images?
        metallicity = np.array([-.96, -.87, -.78, -.69, -.60])
        reddening = np.array([0])
        mag_bins_file = "08ha_bins.npy" # Filename of .npy array w/ edges of magnitude histogram (2D: 1st index=filter, 2nd index=bin)
        mag_frac_file = "08ha_frac.npy" # Filename of .npy array w/ recovery fraction of magnitudes above (same shape as above, but 1 shorter in 2nd dim.)
        new_dir = "MC_08ha_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
    if str(sys.argv[1]) == '10ae':
        print("Running with SN 2010ae parameters.")
        nstars = 11 # Normally is 28, just testing with new dolphot output
        distance = 11.0873e6
        distance_error = 1.02266e6
        filters = ["F435W", "F555W", "F625W", "F814W"]
        F435W_ext = .509 # extinction in F435W in ESO 162-17 from NED
        F555W_ext = .394 # extinction in F555W in ESO 162-17 from NED
        F625W_ext = .313 # extinction in F625W in ESO 162-17 from NED
        F814W_ext = .215 # extinction in F814W in ESO 162-17 from NED
        Zeropoints = np.array([25.0164, 24.9694, 24.9904, 24.7724]) + np.log10(2)
        Exposure_times = 1.9 * np.array([1402, 1002, 1078, 1478])
        WFC3 = False # Are there WFC3 images?
        metallicity = np.array([-.55, -.49, -.43, -.37, -.31])
        reddening = np.array([.08, .22, .36, .50, .64, .78, .92])
        new_dir = "MC_10ae_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
    if str(sys.argv[1]) == '10el':
        print("Running with SN 2010el parameters.")
        nstars = 49
        distance = 5.63e6
        distance_error = 1.09e6
        filters = ["F435W", "F555W", "F625W", "F814W"]
        F435W_ext = 0.033 # extinction in F435W in NGC 1566 from NED
        F555W_ext = 0.025 # extinction in F555W in NGC 1566 from NED
        F625W_ext = .021 # extinction in F625W in NGC 1566 from NED
        F814W_ext = .014 # extinction in F814W in NGC 1566 from NED
        Zeropoints = np.array([25.769, 25.722, 25.743, 25.525])
        Exposure_times = 1.7 * np.array([1168, 768, 844, 1244])
        WFC3 = False # Are there WFC3 images?
        metallicity = np.array([.02, .09, .16, .23, .30])
        reddening = np.array([0, .10254, .20507])
        new_dir = "MC_10el_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
    if (str(sys.argv[1]) == '12z') or (str(sys.argv[1]) == '12Z'):
        print("Running with SN 2012Z parameters.")
        nstars = 9
        distance = 31.96e6
        distance_error = .81e6
        filters = ["F435W", "F555W", "F814W"]
        F435W_ext = 0.144 # extinction in F435W in NGC 1566 from NED
        F555W_ext = 0.112 # extinction in F555W in NGC 1566 from NED
        F814W_ext = 0.061 # extinction in F814W in NGC 1566 from NED
        WFC3 = False # Are there WFC3 images?
        metallicity = np.array([-.59, -.51, -.43, -.35, -.27])
        reddening = np.array([.01, .04, .07, .11, .14])
        new_dir = "MC_12Z_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
    if (str(sys.argv[1]) == '14ck'):
        print("Running with SN 2014ck parameters.")
        nstars = 20 # Was 14
        distance = 24.32e6
        distance_error = 1.69e6
        filters = ["WFC3_F625W", "WFC3_F814W"]
        F625W_ext = 1.018
        F814W_ext = .700
        Zeropoints = np.array([25.396, 24.712])
        Exposure_times = 180 * np.array([510, 680])
        WFC3 = True # Are there WFC3 images?
        metallicity = np.array([-1.03, -.77, -.51, -.25, .01])
        reddening = np.array([0, .025, .05])
        new_dir = "MC_14ck_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
    if str(sys.argv[1]) == '06my':
        print("Running with SN 2006my parameters.")
        nstars = 16 # Normally is 15, just testing with new dolphot output
        distance = 12.2e6
        distance_error = 1.36e6
        filters = ["F555W", "F814W"]
        F555W_ext = 0.075 # extinction in F555W in NGC 4651 from NED
        F814W_ext = 0.041 # extinction in F814W in NGC 4651 from NED
        Zeropoints = np.array([24.9694, 24.7724]) + np.log10(2)
        Exposure_times = 1.65 * np.array([1090, 1090])
        WFC3 = False # Are there WFC3 images?
        metallicity = np.array([-.40, -.29, -.18, -.07, .04])
        reddening = np.array([0])
        mag_bins_file = "06my_bins.npy" # Filename of .npy array w/ edges of magnitude histogram (2D: 1st index=filter, 2nd index=bin)
        mag_frac_file = "06my_frac.npy" # Filename of .npy array w/ recovery fraction of magnitudes above (same shape as above, but 1 shorter in 2nd dim.)
        new_dir = "MC_06my_MagFit_{date}".format(date=np.round(time.time())) # Sets up directory for saving into
        os.makedirs(new_dir)
else: # If no arguments given, uses the arguments for SN 2008ha
    print("Please Give a Command line argument specifying the SN of interest")
    sys.exit()


#################### Sets Up Variables for Pulling From Isochrones ######################

# Checks operating system, to adjust filesystem to work on both.
if platform.system() == "Darwin":
    mist_dir = "/Users/tktakaro/Documents/Type-Iax-HST/MIST/ACS-WFC"
    mist_dir_WFC3 = "/Users/tktakaro/Documents/Type-Iax-HST/MIST/WFC3-UVIS"
elif platform.system() == "Windows":
    mist_dir = "D:/Research/Type-Iax-HST/MIST/ACS-WFC"
    mist_dir_WFC3 = "D:/Research/Type-Iax-HST/MIST/WFC3-UVIS"
else:
    mist_dir = "/data/ttakaro/Type-Iax-HST/MIST/ACS-WFC"
    mist_dir_WFC3 = "/data/ttakaro/Type-Iax-HST/MIST/WFC3-UVIS"
if WFC3:
    kwargs = {"names": ["EEP", "log10_isochrone_age_yr", "initial_mass", "star_mass", "log_Teff", "log_g", "log_L",
                        "[Fe/H]_init", "[Fe/H]", "WFC3_UVIS_F200LP", "WFC3_UVIS_F218W", "WFC3_UVIS_F225W", "WFC3_UVIS_F275W",
                        "WFC3_UVIS_F280N", "WFC3_UVIS_F300X", "WFC3_UVIS_F336W", "WFC3_UVIS_F343N",
                        "WFC3_UVIS_F350LP", "WFC3_UVIS_F373N", "WFC3_UVIS_F390M", "WFC3_UVIS_F390W",
                        "WFC3_UVIS_F395N", "WFC3_UVIS_F410M", "WFC3_UVIS_F438W", "WFC3_UVIS_F467M",
                        "WFC3_UVIS_F469N", "WFC3_UVIS_F475W", "WFC3_UVIS_F475X", "WFC3_UVIS_F487N",
                        "WFC3_UVIS_F502N", "WFC3_UVIS_F547M", "WFC3_UVIS_F555W", "WFC3_UVIS_F600LP",
                        "WFC3_UVIS_F606W", "WFC3_UVIS_F621M", "WFC3_UVIS_F625W", "WFC3_UVIS_F631N",
                        "WFC3_UVIS_F645N", "WFC3_UVIS_F656N", "WFC3_UVIS_F657N", "WFC3_UVIS_F658N",
                        "WFC3_UVIS_F665N", "WFC3_UVIS_F673N", "WFC3_UVIS_F680N", "WFC3_UVIS_F689M",
                        "WFC3_UVIS_F763M", "WFC3_UVIS_F775W", "WFC3_UVIS_F814W", "WFC3_UVIS_F845M",
                        "WFC3_UVIS_F850LP", "WFC3_UVIS_F953N", "WFC3_IR_F098M", "WFC3_IR_F105W", "WFC3_IR_F110W",
                        "WFC3_IR_F125W", "WFC3_IR_F126N", "WFC3_IR_F127M", "WFC3_IR_F128N", "WFC3_IR_F130N",
                        "WFC3_IR_F132N", "WFC3_IR_F139M", "WFC3_IR_F140W", "WFC3_IR_F153M", "WFC3_IR_F160W",
                        "WFC3_IR_F164N", "WFC3_IR_F167N", "phase"],
            "delim_whitespace": True, "comment": "#"}
    mist_dir = mist_dir_WFC3
else:
    kwargs = {"names": ["EEP", "log10_isochrone_age_yr", "initial_mass", "star_mass", "log_Teff", "log_g",
                        "log_L", "[Fe/H]_init", "[Fe/H]", "ACS_WFC_F435W", "ACS_WFC_F475W", "ACS_WFC_F502N",
                        "ACS_WFC_F550M", "ACS_WFC_F555W", "ACS_WFC_F606W", "ACS_WFC_F625W", "ACS_WFC_F658N",
                        "ACS_WFC_F660N", "ACS_WFC_F775W", "ACS_WFC_F814W", "ACS_WFC_F850LP", "ACS_WFC_F892N", "phase"],
            "delim_whitespace": True, "comment": "#"}


isochrones = {}
for filename in glob.glob(mist_dir + "/*.iso.cmd"):
    filename = filename.replace("\\", "/")
    feh_string = filename.split("/")[-1].split("=")[1].split(".iso")[0] # Pulls metalicity information
    feh = float(feh_string) # feh is [Fe/H]
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
def Random_mass_mag(mass, mags):
    m = invSalpeter(np.random.random(), 4, np.amax(mass))
 
    if str(sys.argv[1]) == '08ha':
        mag4, mag5, mag6, mag8 = mags
        # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
        m_index = np.argmin(np.abs(m - mass))
        mags = np.array([mag4[m_index], mag5[m_index], mag6[m_index], mag8[m_index]])
        loc = Exposure_times * 10**((Zeropoints - mags)/2.5)
        scale = np.array([73 + np.sqrt(loc[0]), 71 + np.sqrt(loc[1]), 86 + np.sqrt(loc[2]), 130 + np.sqrt(loc[3])])
    elif str(sys.argv[1]) == '10ae':
        mag4, mag5, mag6, mag8 = mags
        # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
        m_index = np.argmin(np.abs(m - mass))
        mags = np.array([mag4[m_index], mag5[m_index], mag6[m_index], mag8[m_index]])
        loc = Exposure_times * 10**((Zeropoints - mags)/2.5)
        scale=np.array([125 + 1.4*np.sqrt(loc[0]), 125 + 1.4*np.sqrt(loc[1]), 160 + 1.4*np.sqrt(loc[2]),
                        250 + 1.5*np.sqrt(loc[3])])
    elif str(sys.argv[1]) == '10el':
        mag4, mag5, mag6, mag8 = mags
        # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
        m_index = np.argmin(np.abs(m - mass))
        mags = np.array([mag4[m_index], mag5[m_index], mag6[m_index], mag8[m_index]])
        loc = Exposure_times * 10**((Zeropoints - mags)/2.5)
        scale = np.array([120 + np.sqrt(loc[0]), 125 + np.sqrt(loc[1]), 160 + np.sqrt(loc[2]), 255 + np.sqrt(loc[3])])
    elif (str(sys.argv[1]) == '12z') or (str(sys.argv[1]) == '12Z'):
        mag4, mag5, mag8 = mags
        # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
        loc=np.array([10**(-.4 * mag4[np.argmin(np.abs(m - mass))]), 10**(-.4 * mag5[np.argmin(np.abs(m - mass))]), 
                      10**(-.4 * mag8[np.argmin(np.abs(m - mass))])])
        scale=np.array([5.4e-13 + 1e-7*np.sqrt(loc[0]), 5.3e-13 + 1.24e-7*np.sqrt(loc[1]), 5e-13 + 2e-7*np.sqrt(loc[2])])
    elif str(sys.argv[1]) == '14ck':
        mag6, mag8 = mags
        # Determines the magnitude corresponding to the point on the isochrone closest in mass to the chosen mass
        m_index = np.argmin(np.abs(m - mass))
        mags = np.array([mag6[m_index], mag8[m_index]])
        loc = Exposure_times * 10**((Zeropoints - mags)/2.5)
        scale = np.array([59 + np.sqrt(loc[0]), 75 + np.sqrt(loc[1])])
    fluxs = np.random.normal(loc=loc, scale=scale, size=len(mags))
    # Computes signal to noise ratio, to be used in selecting observable stars. SN is overall S/N ratio
    N = np.sqrt(1/np.sum(np.power(scale, -2)))
    SN = N * np.sum(fluxs/np.power(scale, 2))
    # SN2 is the maximum S/N ratio in a single band
    SN2 = np.amax(fluxs/scale)

    return [m, SN, SN2, fluxs]


""" This function generates a set of false stars using the errors in magnitude and distance, assuming normal
    distributions. It then performs the Crappy Chi-squared (CChi) statistical test, in order to compare to real
    stars. Some notes: 1. It weights by physical distance, but then divides out the average weight at the end, in
    order to avoid skewing a comparison with real stars, where the average weight may be different. 2. This function
    is written with bad coding practices, as it uses several variables defined outside the function. For this reason,
    be very careful (just don't) using this function outside of the bounds of this script.
"""
def False_Stars_CChi(reddening, age):
    global cont
    if cont == False:
        return np.inf
    np.random.seed()
    # Chooses distance using gaussian with errors from literature
    dist = np.random.normal(loc=distance, scale=distance_error)
    dist_adjust = 5 * (np.log10(dist) - 1) # Converts distance to a magnitude adjustment
    flat = (100 * 206265)/(dist * .05) # 100 parsecs in pixels
    flat_int = int(np.round(flat*5))
    while (flat_int < 0) or (flat_int >= 2000): # Checks to make sure distance isn't so crazy it'll cause errors
            dist = np.random.normal(loc=distance, scale=distance_error)
            dist_adjust = 5 * (np.log10(dist) - 1)
            a = flat_int
            flat = (100 * 206265)/(dist * .05)
            flat_int = int(np.round(5 * flat))
            print("Was ", a, "but is now ", flat_int)

    idx = df.log10_isochrone_age_yr == age
    mass = df[idx].initial_mass

    mags = []
    if "F435W" in filters:
        mag_435 = df[idx].ACS_WFC_F435W + dist_adjust + F435W_ext + 3.610*reddening
        mags.append(mag_435)
    if "F555W" in filters:
        mag_555 = df[idx].ACS_WFC_F555W + dist_adjust + F555W_ext + 2.792*reddening
        mags.append(mag_555)
    if "F625W" in filters:
        mag_625 = df[idx].ACS_WFC_F625W + dist_adjust + F625W_ext + 2.219*reddening
        mags.append(mag_625)
    if "F814W" in filters:
        mag_814 = df[idx].ACS_WFC_F814W + dist_adjust + F814W_ext + 1.526*reddening
        mags.append(mag_814)
    if "WFC3_F625W" in filters:
        mag_WFC3_625 = df[idx].WFC3_UVIS_F625W + dist_adjust + F625W_ext + 2.259*reddening
        mags.append(mag_WFC3_625)
    if "WFC3_F814W" in filters:
        mag_WFC3_814 = df[idx].WFC3_UVIS_F814W + dist_adjust + F814W_ext + 1.436*reddening
        mags.append(mag_WFC3_814)

    # Load in recovery fraction arrays
    mag_bins = np.load(mag_bins_file)
    mag_frac = np.load(mag_frac_file)

    # This array will hold 1. Mass 2. Radial distance 3+. Magnitudes
    False_stars = np.zeros([nstars, 2 + len(mags)])

    temp = 0 # This will hold the cumulative difference in magnitude between the stars and isochrone
    phys_dist_temp = 0 # This will hold the comulative phyical distance between the stars and the SN position
    for x in range(nstars):
        # Generates stars with randomly drawn mass, then finds corresponding flux in each filter
        False_stars[x,0], SN, SN2, fluxs = Random_mass_mag(mass, mags)
        False_stars[x,2:] = fluxs

        # Applies recovery fraction to magnitudes generated
        frac_temp = np.zeros(len(mags))
        while not any(np.random.random() < frac_temp):
            # Checks to make sure that the S/N ratio is high enough, and there is positive flux in each filter
            t = time.time()
            while (SN < 3.5) or (SN2 < 2.5) or any(False_stars[x,2:] < 0):
                False_stars[x,0], SN, SN2, fluxs = Random_mass_mag(mass, mags)
                False_stars[x,2:] = fluxs
                if time.time() - t > 10:
                    cont = False
                    return np.inf

            # Applies the recovery fraction to generated magnitudes
            mag_temp = Zeropoints - 2.5 * np.log10(False_stars[x,2:]/Exposure_times) # Calculates mags from fluxs
            for y in range(len(mags)): # Checks each filter
                mag_index = np.argmin(np.abs(mag_temp[y] - (mag_bins[y][:-1] + mag_bins[y][1:])/2))
                frac_temp[y] = mag_frac[y][mag_index]

        # Converts from flux to magnitude in each filter
        False_stars[x,2:] = Zeropoints - 2.5 * np.log10(False_stars[x,2:]/Exposure_times)
    
        # Samples radial distribution to get radial distance from SN
        sigma = 5 * (.92 * 10**age * 3.15e7 * 206265)/(dist * 3.086e13 * .05) # 5 times, as weight_func is spaced with 5 spots per pixel
        # Adds in inherent spread in star position at formation with the of 100 parsecs
        False_stars[x,1] = abs(np.random.normal(loc=0, scale=sigma)) + flat * np.random.random()
    
        # Now, determine Crappy Chi-squared fit
        # Convolves a normal distribution with a flat distribution to get distribution used above to generate radius
        weight_func = np.convolve(1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-np.linspace(-200,200,2000)**2/(2 * sigma**2)),
               np.append(np.zeros(int(np.ceil((2000-flat_int)/2))),np.append(np.ones(flat_int),np.zeros(int(np.floor((2000-flat_int)/2))))))
        # Finds where in the convolved array the generated radius falls
        try: phys_dist_weight = weight_func[1999 + int(False_stars[x,1]*5)]
        except IndexError: phys_dist_weight = weight_func[weight_func.size - 1]
        phys_dist_temp += phys_dist_weight # Will be used to compute average of the weights

        # Adds the magnitude difference for each data point in quadrature.
        diff_tot = np.zeros(mags[0].shape)
        for i in range(len(mags)):
            diff_tot += (False_stars[x,2+i] - mags[i])**2
        temp += phys_dist_weight * np.amin(np.sqrt(diff_tot))
    phys_dist_temp /= nstars
    return np.sqrt(temp)/phys_dist_temp


######################## Runs the Monte Carlo ########################
# 1st index: metallicity, 2nd index: reddening, 3rd index: age, 4th index: metallicity, reddening, age, or CChi value
CChi_false = np.zeros([metallicity.size, reddening.size, 41, 4, 5000])
CChi = np.zeros([2,5000])

# Generates false stars and applies a CChi test 1000 times to get a distribution of values
cont = True # Variable used for halting when generating stars takes too long

outtext = "{Dir}/Info.txt".format(Dir=new_dir) # Saves the reddening, extinction info
np.save(outtext, ["Metallicity = {Metal}, Reddening = {Redden}".format(Metal=metallicity, Redden=reddening)])

for k, met in enumerate(metallicity):
    df = isochrones[met] # Sets metallicity. Eventually this will be varied over.
    ages = np.array(list(set(df.log10_isochrone_age_yr)))
    ages.sort()
    age_cmd = {}
    ages = ages[(ages > 6.49) & (ages < 8.51)] # Sets ages to consider.
    for j, red in enumerate(reddening):
        for i, age in enumerate(ages):
            CChi_false[k, j, i, 1, :] = met
            CChi_false[k, j, i, 2, :] = red
            CChi_false[k, j, i, 3, :] = age
            func = partial(False_Stars_CChi, red) # Turns False_Stars into a single parameter function
            if __name__ == '__main__':
                pool = mp.Pool(os.cpu_count() - 2)
                print("Working on age={Age}, metal={Met}, reddening={Red}".format(Age=np.round(age,decimals=2), Met=met, Red=red))
                results = pool.map_async(func, age * np.ones(5000)).get()
                CChi[1,:] = list(results)
                pool.close()
            CChi_false[k,j,i,0,:] = CChi[1,:]
        outfile = "{Dir}/CChi_false_{in1}{in2}".format(Dir=new_dir, in1=k, in2=j)
        np.save(outfile, CChi_false)

outfile = "{Dir}/CChi_false_ages".format(Dir=new_dir) # Saves all ages together
np.save(outfile, CChi_false)
