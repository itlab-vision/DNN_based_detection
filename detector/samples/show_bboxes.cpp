#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "show_bboxes <file_name>" << endl
             << "\t<file_name> - file name that contained bboxes list" << endl;
        return 0;
    }
    char *fileName = argv[1];
    ifstream file(fileName);
    if (!file.is_open())
    {
        cout << "Error: file was not opened." << endl;
        return 0;
    }
    
    string imgName;
    file >> imgName;
    cout << imgName << endl;
    Mat img = imread(imgName, 1);
    int kBboxes;
    file >> kBboxes;
    for (int i = 0; i < kBboxes; i++)
    {
        Rect bbox;
        float score;
        file >> bbox.x >> bbox.y >> bbox.width >> bbox.height >> score;
        rectangle(img, bbox, Scalar(255, 0, 0), 2);
    }
    namedWindow("Bounding boxes");
    imshow("Bounding boxes", img);
    waitKey();

    file.close();
    
    return 1;
}