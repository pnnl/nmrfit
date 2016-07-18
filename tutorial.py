import NMRfit as nmrft
import os
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


class Tutorial:
    def __init__(self, outF):
        # # for user test
        # self.inDir = "./Data/organophosphate/"
        # self.datasets = ["dc-919V_cdcl3_kilimanjaro_25c_1d_1H_1_031916.fid"] * 10

        # for blinded evaluation
        self.inDir = "./Data/blindedData"
        self.datasets = [outF[7:-4]] * 3

        self.outF = open(outF, 'w')
        self.dataIndex = 0
        self.failedAttempts = 0
        self.firstTimePrompt()
        while self.dataIndex < len(self.datasets):
            self.run(self.datasets[self.dataIndex])
        self.outF.close()
        print('Done!')

    def firstTimePrompt(self):
        response = input("Welcome to the exciting new NMR fitting utility!  Is this your first time? (y/n) ")
        response = response.strip()
        response = response.lower()

        if response == 'y' or response == 'yes':
            print("Great, glad to have you! I'll make sure you the process is as smooth as possible.")
            self.firstTime = True
        if response == 'n' or response == 'no':
            print("Good to know.  I'll spare you some details then.")
            self.firstTime = False
        print()

    def boundsSelectionPrompt(self):
        if self.firstTime is True:
            print("Your first task is to select the appropriate bounds of the data.  I'm going to open up a plot, "
                  "and you need to click on either side of the region of interest (ROI).")
            print("But what's the ROI?")
            print("Look for two large peaks in the spectroscopy data.  On either side of these peaks, you'll notice "
                  "several smaller peaks.  These are called satellite peaks.  We are interested in ONE set of satellite "
                  "peaks on either side of the main peaks (that is, the data should include two small peaks per side, "
                  "for a total of four small peaks).")
            print("Click on the graph to bound the data--once on the left, once on the right--such that the main peaks "
                  "and four satellite peaks are captured.  Try to get the main peaks in the center of your bounds. "
                  "Note that order of your clicks does not matter.")

        else:
            print("Select the appropriate bounds of the region of interest.")

        input("Press enter when ready.")
        print()

    def peakSelectionPrompt(self):
        if self.firstTime is True:
            print("Next I need you to select each of the peaks.  You need to click the graph three times "
                  "for each peak: once on it's tip, and once on either side when the data just starts to flatten "
                  "out.  The order for each peak (left, center, right) does not matter, but I want you to start with "
                  "the main  peaks, then move on to the smaller ones.")

        else:
            print("Starting with the main peaks, select each peak's center, left and right.")

        input("Press enter when ready.")
        print()

    def fittingPrompt(self):
        if self.firstTime is True:
            print("The fitting algorithm will now run.  This takes some time, so be patient.")
        else:
            print("Data is now being fit.")

    def fitEvaluation(self):
        if self.firstTime is True:
            print("The code is pretty sensitive to initial conditions.  Sometimes, the fit misses the smaller peaks. "
                  "I am going to show you the results, and your job is to assess whether all the peaks were fit.")
        else:
            print("Results ready.")
        input("Press enter when ready.  Simply close the plot when finished.")
        print()

    def fitEvaluation2(self):
        response = input("Were all peaks successully fit? (y/n) ")
        response = response.strip()
        response = response.lower()

        if response == 'y' or response == 'yes':
            nLeft = len(self.datasets) - self.dataIndex - 1
            if nLeft > 0:
                print("Great! We still have", str(nLeft), "more datasets for you to try.\n")
            return True
        if response == 'n' or response == 'no':
            print("It happens.  I'm going to have you try again.\n")

            return False

    def futurePrompts(self):
        if self.firstTime is True:
            response = input("Would you like to continue with detailed prompts(y/n)? ")
            response = response.strip()
            response = response.lower()

            if response == 'y' or response == 'yes':
                self.firstTime = True
            if response == 'n' or response == 'no':
                self.firstTime = False
            print()

    def run(self, dataset):
        # Load and process data
        path = os.path.join(self.inDir, dataset)
        w, u, v, theta0 = nmrft.varian_process(os.path.join(path, 'fid'), os.path.join(path, 'procpar'))

        # create data object
        data = nmrft.Data(w, u, v, theta0)

        self.boundsSelectionPrompt()

        # initiate bounds selection
        data.select_bounds()

        self.peakSelectionPrompt()

        # interactively select peaks and satellites
        p = data.select_peaks(2)
        s = data.select_satellites(4)

        # weights across ROIs
        weights = [['all', 1.0],
                   ['all', 1.0],
                   [s[0].bounds, p[0].height / s[0].height],
                   [s[1].bounds, p[0].height / s[1].height],
                   [s[2].bounds, p[0].height / s[2].height],
                   [s[3].bounds, p[0].height / s[3].height]]

        # initial conditions of the form [theta, r, yOff, sigma_n, mu_n, a_n,...]
        x0 = [0, 0.1, 0,                                  # shared params
              p[0].width / 10, p[0].loc, p[0].area,       # p1 init
              p[1].width / 10, p[1].loc, p[1].area,       # p2 init
              p[1].width / 10, s[0].loc, s[0].area,       # s1 init
              p[1].width / 10, s[1].loc, s[1].area,       # s2 init
              p[1].width / 10, s[2].loc, s[2].area,       # s3 init
              p[1].width / 10, s[3].loc, s[3].area]       # s4 init

        self.fittingPrompt()

        # fit data
        fit = nmrft.FitUtility(data, x0, weights=weights)

        # generate result
        res = fit.generate_result(scale=1)

        plt.plot(data.w, data.v)
        plt.plot(res.w, res.v)
        plt.show()

        percent = (res.params[11] + res.params[14] + res.params[17] + res.params[20]) / (res.params[5] + res.params[8] + res.params[11] + res.params[14] + res.params[17] + res.params[20])

        success = self.fitEvaluation2()
        if success is True:
            self.outF.write('Dataset: ' + self.datasets[self.dataIndex] + '\n')
            self.outF.write('Failed attempts: ' + str(self.failedAttempts) + '\n')
            self.failedAttempts = 0
            self.outF.write('Percent: ' + str(percent) + '\n')
            self.outF.write('SSD: ' + str(res.error) + '\n')

            out = ''
            for j in res.params:
                out += ' ' + str(j)
            self.outF.write('Fit params:' + out + '\n')
            self.outF.write('\n')
            self.dataIndex += 1
        else:
            self.failedAttempts += 1

        self.futurePrompts()

if __name__ == '__main__':
    t = Tutorial('./sean_dc_4a_cdcl3_kilimanjaro_25c_1d_1H_3_070916.fid.txt')
