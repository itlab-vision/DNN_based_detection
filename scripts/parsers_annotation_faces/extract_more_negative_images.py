import cv2 as cv
import time
import sys

def pyramid(image, scale=1.5, minSize=(20, 20)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image =cv.resize(image, (w,h))
        #print( " new size image " )
        #print(image.shape)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def image_sliding_windows(image,name,file,path,
                          (winW,winH)= (108,108)):
    scale=1.2
    count=0
    for resized in pyramid(image, scale):
        for (x, y, window) in sliding_window(resized, stepSize=6, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            count += 1
            cv.imwrite(path + name + "-" + count.__str__() + ".jpg",  cv.resize(resized[int(y):int(winH)+int(y),int(x):int(winW)+ int(x)], (227,227)))
            file.write(path +  name+ "-"  + count.__str__() + ".jpg")



def main(list_negative, path_to_reshaped, negative_images ):
    _new_image_list = open(list_negative,'w')
    _file = open(negative_images,'r')
    count = 0

    for line in _file:
	name = line.split()[0]
	print ("../" + name.__str__() + " start")
	start = time.time()
	target_image = cv.imread("../" + name )
	crop_img = cv.resize(target_image, (227, 227))
	cv.imwrite(path_to_reshaped + count.__str__() + ".jpg", crop_img)
	_new_image_list.write(path_to_reshaped + name.__str__()  + "\n")
	name = name[0: name.index(".",0,len(name))]
	image_sliding_windows(target_image, count.__str__(),_new_image_list,path_to_reshaped)
	print("end " + name.__str__() + " time=" + (time.time() - start).__str__())       
	count = count + 1 
    _new_image_list.close()

if __name__ == '__main__':
     if  ( len(sys.argv) ==4):
	    main(sys.argv[1],sys.argv[2],sys.argv[3])


"""
  _new_image_list = open("list_negative.txt",'w')
    path_to_reshaped = '/home/maljutina_e/DataSet_for_train_Faces_Detector/all/'
    _file = open('negative_linux.txt','r')
"""
