
"""
remove replay in annotation file


_path_old_annot = "D:/my-study/OpenCV/DataSet/GENKI-R2009a/Subsets/GENKI-SZSL/annot.txt"
_path_new_annot"D:/my-study/OpenCV/DataSet/DataSet_for_train_Faces_Detector/PositiveImage/annotation.txt"
_path = "DataSet_for_train_Faces_Detector/PositiveImage/GENKI-R2009a/files/"
"""
import sys

def parse(_path_old_annot,_path_new_annot,_path):
	file = open(_path_old_annot)
	annotFile = open(_path_new_annot,'a')
	count = 1
	flag = 0
	listName = []
	listNameNew = []
	for line in file:
		listName.append(line)
		if listName.__len__() >= 2:
			one = listName[listName.__len__()-1].split()
			two = listName[listName.__len__()-2].split()
			if one[0]!=two[0]:
				if flag==1:
					listName.remove(line)
					count=0
					annotFile.write(_path+two[0]+" ")	 
					for str in listName:
						count+=1
					annotFile.write(count.__str__())
					for str in listName:
						s= str.split()
						annotFile.write(" "+s[2]+" "+s[3]+" " + s[4]+" " + s[5])
					annotFile.write("\n")
				else:
					annotFile.write("DataSet_for_train_Faces_Detector/PositiveImage/GENKI-R2009a/files/"+ two[0] + " "+two[1] + " "+two[2] + " "+two[3] + " "+two[4] + " "+two[5] + "\n")
				listName=[]
				listName.append(line)
				flag=0
			else:
				flag=1
	annotFile.close()

if __name__== "__main__":
   _path_old_annot,_path_new_annot,_path = sys.argv[1:4]
    parse(_path_old_annot,_path_new_annot,_path)