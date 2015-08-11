

'''
file  = open ("D:/my-study/OpenCV/DataSet/MUCT/muct-landmarks/muct76-opencv.csv")

annot = open ("D:/my-study/OpenCV/DataSet/DataSet_for_train_Faces_Detector/PositiveImage/annotation.txt",'a')
_path = "DataSet_for_train_Faces_Detector/PositiveImage/MUCT/jpg/"

'''


def parse(file,annot,_path):
    c=0
    for line in file:
        if c>0:

            maxY = 0
            minY = 1000
            maxX = 0
            minX = 1000

            mass = line.split(",")

            for i in range(mass.__len__()):

                if i==0:
                    name =_path  +mass[i] + ".jpg"

                if i>1:
                    if  i % 2 ==0:
                        if maxX< float(mass[i]):
                            maxX=float(mass[i])
                        if minX > float(mass[i]) and float(mass[i])!=0.0:
                            minX = float(mass[i])
                    else:
                        if maxY < float(mass[i]):
                            maxY = float(mass[i])
                        if minY > float(mass[i]) and float(mass[i])!=0.0:
                            minY = float(mass[i])

           # print(minX)
            h = maxY - minY + 0.2*(maxY - minY)
            w = maxX - minX  + 10
            x = minX - 5
            y = minY - h*0.2
            annot.write(name + " " + "1 " + x.__str__()  + " " + y.__str__() + " " + w.__str__() + " " + h.__str__() + "\n")
        c+=1

    annot.close()

if __name__== "__main__":
    annot, file, _path = sys.argv[1:4]
    parse(file,annot,_path)