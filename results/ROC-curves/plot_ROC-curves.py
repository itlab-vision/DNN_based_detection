import pylab
import argparse
import matplotlib.pyplot as plt
import math
from itertools import cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required = True, help = "path to  list ")
    print(parser)
    args = vars(parser.parse_args())
    print (args)
    file = args["file"]
    dirName = open(file,'r')
    plt.figure("DiscRoc")
    plt.xlabel("False Positive Per Frame")
    plt.ylabel("True Positive Rate")
    plt.grid()
    listX=[]
    listY=[]
    listLeg=[]
    line_styles = ['-', '--', ':', '-.']
    line_style_cycler =  cycle( [s for s in line_styles for _ in plt.rcParams['axes.color_cycle'] ])
    for dirLine in dirName:
        file = open(dirLine [0: len(dirLine)-1])
        for line in file:
                name = line.split()
                if (float(name[1]) < 100000):
                    listX.append(int(name[1])/float(2844))
                    listY.append(name[0])
        if (dirLine.find("LBP")!=-1):
            if (dirLine.find("OpenCV") != -1):
                listLeg.append( dirLine[ dirLine.rfind("LBP"): len(dirLine)-13] )
            else:
                listLeg.append(dirLine [dirLine.find("OpenCV"): len(dirLine)-1])
        if (dirLine.find("Haar")!=-1):
            if (dirLine.find("OpenCV") != -1):
                listLeg.append(dirLine [dirLine.rfind("haar"): len(dirLine)-13])
            else:
                listLeg.append(dirLine [dirLine.find("OpenCV"): len(dirLine)-1])
        plt.plot(listX,listY,linestyle=next(line_style_cycler))
        listY=[]
        listX=[]
    plt.legend(listLeg, prop={'size':9})
    plt.show()
    
#=====================================
if __name__ == '__main__':
    main()