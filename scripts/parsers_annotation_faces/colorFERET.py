
'''
name = "D:/my-study/OpenCV/DataSet/colorferet/name"
annotation = open("D:/my-study/OpenCV/DataSet/DataSet_for_train_Faces_Detector/PositiveImage/annotation.txt",'a')
nameDir = "D:/my-study/OpenCV/DataSet/colorferet/dvd"
_path = "DataSet_for_train_Faces_Detector/PositiveImage/colorferet/image/"
_add_path_ = "/data/ground_truths/name_value/"
'''

import sys

def parse(annotatioin,name,nameDir,_add_path):
    for i in range(2):
        i+=1
        nameFile = name + i.__str__()
        nameAnnotDir = nameDir + i.__str__()+ _add_path_
        file = open(nameFile + ".txt")
        count=0
        for line in file:
            line = line.split()
            if line.__len__()>3 :
                if line[0].__len__()>1 and line[3].__len__()>9:
                    nameImage = _path + line[3][0: line[3].__len__() - 4] + ".jpg"
                    annotFile = nameAnnotDir + line[3][0:5] + "/" + line[3]
                    annot  = open(annotFile)
                    listPoint=[]
                    for annotLine in annot:
                        annotLine = annotLine.split()
                        if annotLine.__len__()>1 and annotLine.__len__()<3:
                            if annotLine[0].find("left_eye_coordinates=")!=-1:
                                listPoint.append(float(annotLine[0][21:]))
                            if annotLine[0].find("right_eye_coordinates=")!=-1:
                                listPoint.append(float(annotLine[0][22:]))
                            if annotLine[0].find("nose_coordinates=")!=-1:
                                listPoint.append(float(annotLine[0][17:]))
                            if annotLine[0].find("mouth_coordinates=")!=-1:
                                listPoint.append(float(annotLine[0][18:]))
                            listPoint.append(float(annotLine[1]))
                        if listPoint.__len__()==8:
                            w = (listPoint[0] - listPoint[2])*2
                            h = (listPoint[7] - listPoint[5])*5.5
                            y = abs(listPoint[3] - h*0.35)
                            x = listPoint[2]- w*0.25
                            annotation.writelines(nameImage + " "+"1 "+x.__str__()+" "+y.__str__()+" " + w.__str__() + " " +h.__str__() )
                            listPoint=[]
            count+=1
    annotation.close()

if __name__== "__main__":
    annot, name, NameDir, _add_path = sys.argv[1:5]
    parse(annot,name,nameDir,_add_path)
