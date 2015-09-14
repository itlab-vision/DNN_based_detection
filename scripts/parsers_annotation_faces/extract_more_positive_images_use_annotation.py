
import cv2 as cv
import sys
import time


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

def image_sliding_windows(image,xr,yr,wr,hr,name,file,path,
                          (winW,winH)= (108,108)):
    scale=1.2

    it = False
    count=0
    for resized in pyramid(image, scale):
        if (it):
            xr,yr , wr,hr = int( int(xr) / scale), int(int(yr) / scale),int(int(wr) / scale),int(int(hr) / scale)
        it = True
        area_annot = int(wr)*int(hr)

        for (x, y, window) in sliding_window(resized, stepSize=6, windowSize=(winW, winH)):

            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            count += 1
            if (y<=yr):
                if (x<xr):
                    if (x + winW < xr + wr):
                        if (y+winH < yr + hr):
                            area = (int(x)+ int(winW) - int(xr))*(int(y) + int(winH) -int(yr))
                        else:
                            area = (x+winW - xr)*( hr)
                    else:
                        if (y+winH < yr + hr):
                            area = (wr)*(y+winH-yr)
                        else:
                            area = (wr)*(hr)
                else:
                    if (x + winW < xr + wr):
                        if (y+winH < yr + hr):
                            area = (winW)*(y+winH-yr)
                        else:
                            area = (winW)*(hr)
                    else:
                        if (y+winH < yr + hr):
                            area = (xr+wr - x)*(y+winH-yr)
                        else:
                            area = (xr+wr-x)*(hr)
            else:
                if (x<xr):
                    if (x + winW < xr + wr):
                        if (y+winH < yr + hr):
                            area = (x + winW - xr)*(winH)
                        else:
                            area = (x + winW - xr)*(yr+ hr - y )
                    else:
                        if (y+winH < yr + hr):
                            area = (wr)*(winH)
                        else:
                            area = (wr)*(hr + yr - y)
                else:
                    if (x + winW < xr + wr):
                        if (y+winH < yr + hr):
                            area = (winW)*(winH)
                        else:
                            area = (winW)*( yr + hr - y)
                    else:
                        if (y+winH < yr + hr):
                            area = (xr + wr - x)*(winH)
                        else:
                            area = (xr + wr - x)*(yr + hr - y)

            if (float(area) / float(area_annot ) > 0.5):
                print("new name image = " + name + "_" + count.__str__() + ".jpg" )
                cv.imwrite(path + name + "-" + count.__str__() + ".jpg",  cv.resize(resized[int(y):int(winH)+int(y),int(x):int(winW)+ int(x)], (227,227)))
                file.write(path +  name + "_"  + count.__str__() + ".jpg")
                """
                cv.imshow("Window_new ", resized[y:y + winH, x:x + winW])
                clone = resized.copy()
                cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv.putText(clone,int(area).__str__() + "/" + int(area_annot).__str__() + "=" + (float(area)/float(area_annot)).__str__(),(x+15,y+15),cv.FONT_ITALIC,0.3,(255,0,0),1)
                cv.rectangle(clone, (xr,yr), (xr+wr,yr+hr), (0,0,255), 2)
                cv.imshow("Window", clone)
                cv.waitKey(30)
                time.sleep(0.1)
                """


def main(list_positive,  path_to_data ,  path_to_reshaped, annotation ):
    _new_positive_image_list = open( list_positive,'w')
    _file = open(annotation,'a+')

    for line in _file:
        name, x, y, w, h = line.split()
        print ("face " + name.__str__() + " start")
        start = time.time()
        target_image = cv.imread(path_to_data + name )
        crop_img = target_image[int(y):int(h)+int(y),int(x):int(w)+ int(x)]
        crop_img = cv.resize(crop_img, (227, 227))
        cv.imwrite(path_to_reshaped + name, crop_img)
        _new_positive_image_list.write(path_to_reshaped + name + "\n")
        name = name[0: name.index(".",0,len(name))]
        image_sliding_windows(target_image,x,y,w,h,name,_new_positive_image_list,path_to_reshaped)
        print("end face id" + name.__str__() + " time=" + (time.time() - start).__str__())
    _new_positive_image_list.close()

if __name__ == '__main__':
    if (len(sys.argv)==5) :
   	 main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


 ''' _new_positive_image_list = open("list_positive.txt",'w')
    path_to_data = '/home/maljutina_e/aflw/aflw/data/flickr/all/'
    path_to_reshaped = '/home/maljutina_e/aflw/aflw/data/flickr/reshaped/'
    _file = open('annotation.txt','a+')
'''

