
"""
format
0 or 1 - smile
fileAnnot = open ("D:/my-study/OpenCV/DataSet/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_labels.txt")
fileNameImag= open ("D:/my-study/OpenCV/DataSet/GENKI-R2009a/Subsets/GENKI-SZSL/GENKI-SZSL_Images.txt")

annot = open("D:/my-study/OpenCV/DataSet/GENKI-R2009a/Subsets/GENKI-SZSL/annot.txt",'w')

name="D:/my-study/OpenCV/DataSet/GENKI-R2009a/files/"
"""

def parse(fileAnnot,fileNameImag,annot,name):

    for line in fileAnnot:
        nameImg = name + fileNameImag.readline()
        nameImg = nameImg[0:nameImg.__len__()-1]
        line = line.split()
        x = float(line[0]) - float(line[2])*0.55
        y = float(line[1]) - float(line[2])*0.55
        h = float(line[2])*1.25
        w = float(line[2])*1.15
        annot.writelines(nameImg + " " + "1 " + x.__str__()+" "+y.__str__()+ " "+w.__str__()+" "+h.__str__() )

    annot.close()

if __name__== "__main__":
    annot, name, fileAnnot, fileNameImag = sys.argv[1:4]
    parse(fileAnnot,fileNameImag,annot,name)

