

"""
0 - name
1 - x left eyes
2 - y left eyes
3 - x right eye
4 - y right eye
5 - x nose
6 - y nose
7 - x left corner mouth
8 - y left corner mouth
9 - x center mouth
10 - y center mouth
11 - x right corner mouth
12 - y right corner mouth


file = open("D:/my-study/OpenCV/DataSet/CMU/annotTest.txt")

annotFile = open("D:/my-study/OpenCV/DataSet/CMU/annotRect.txt",'w')
"""

def parse(file,annotFile):
    for line in file:
        massLine = line.split()
        annotFile.write(massLine[0]+" ")
        h = 7.5 * (float(massLine[10]) - float(massLine[6]))
        w = 5 * ( (float(massLine[3]) - float(massLine[1]))/2 )
        x = float(massLine[1]) - 1.5 * (w/5)
        y = float(massLine[2]) - h*0.4
        print(massLine[0] + " 1" + " " + x.__str__() + " "+ y.__str__() + " "+ w.__str__() + " " + h.__str__())
        annotFile.writelines("1" + " " + x.__str__() + " "+ y.__str__() + " "+ w.__str__() + " " + h.__str__() + "\n")
        print(massLine)
    annotFile.close()

if __name__== "__main__":
    annot, file = sys.argv[1:3]
    parse(file,annot)

