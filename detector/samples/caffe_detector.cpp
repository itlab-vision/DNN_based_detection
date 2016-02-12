#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "detector.hpp"
#include "classifier_factory.hpp"

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#define CV_TIMER

#if defined(CV_TIMER)
    #include <stdio.h>
    #define TIMER_START(name) int64 t_##name = cv::getTickCount()
    #define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
                1000.f * ((cv::getTickCount() - t_##name) / cv::getTickFrequency()))
#elif defined(MPI_TIMER)
    #include <stdio.h>
    #define TIMER_START(name) double t_##name = MPI_Wtime(); 
    #define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fs\n", MPI_Wtime() - t_##name)
#else
    #define TIMER_START(name)
    #define TIMER_END(name)
#endif

using namespace std;
using cv::Rect;
using cv::Mat;
using cv::Size;
using cv::FileNode;
using cv::FileStorage;

const char * params =
    "{ h | help           | false      | print help message     }"
    "{ c | config         |            | config file            }"
    "{ a | annotation     |            | annotation file        }"
    "{ o | result         | result.txt | file to save result to }";


#if defined(HAVE_MPI) && defined(PAR_SET_IMAGES)

void detect(Detector &detector, const vector<string> &fileNames, const string &outFileName)
{
    int rank, np, fileStep, leftIdx, rigthIdx;
    MPI_Init(0, 0);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    fileStep = (fileNames.size() + np - 1) / np;
    leftIdx = rank * fileStep;
    rigthIdx = min((rank + 1) * fileStep, (int)(fileNames.size()));    
    string fileLine = "";
    for (int i = leftIdx; i < rigthIdx; i++)
    {
        string fileName = fileNames[i];
        Mat img = imread(fileName, cv::IMREAD_COLOR);        
        cout << "Processing " << fileName << endl;
    
        vector<int> labels;
        vector<Rect> rects;
        vector<double> scores;
        TIMER_START(DETECTION);
        detector.DetectMultiScale(img, labels, scores, rects);
        TIMER_END(DETECTION);
        
        fileLine += fileName + "\n" + to_string(rects.size()) + "\n";
        for (size_t j = 0; j < rects.size(); j++)
        {
            fileLine += to_string(rects[j].x) + " " + to_string(rects[j].y) + " "
                + to_string(rects[j].width) + " " + to_string(rects[j].height) + " "
                + to_string(scores[j]) + " \n";
        }
    }
    ofstream res_file(outFileName + to_string(rank) + ".txt");
    res_file << fileLine;
    MPI_Finalize();
}

#elif defined(HAVE_MPI) && defined(PAR_PYRAMID)

void detect(Detector &detector, const string &fileName, const string &outFileName)
{    
    Mat img = imread(fileName, cv::IMREAD_COLOR);
    cout << "Processing " << fileName << endl;
    
    vector<int> labels;
    vector<Rect> rects;
    vector<double> scores;

    TIMER_START(DETECTION);
    detector.DetectMultiScale(img, labels, scores, rects);
    TIMER_END(DETECTION);

    // write to file on 0 process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        ofstream out(outFileName, ios_base::app);
        if (!out.is_open())
        {
            cout << "Failed to open output file\n";
            return;
        }
        out << fileName << endl << rects.size() << endl;
        for (size_t j = 0; j < rects.size(); j++)
        {
            out << rects[j].x << " " << rects[j].y << " "
                << rects[j].width << " " << rects[j].height << " "
                << scores[j] << " " << endl;
        }
        out.close();
    }
}

void detect(Detector &detector, const vector<string> &fileNames, const string &outFileName)
{
    MPI_Init(0, 0);
    // Create/clear output file.
    ofstream outputFile(outFileName);

    for (const auto& filename : fileNames)
    {
        detect(detector, filename, outFileName);
    }
    MPI_Finalize();
}

#else

void detect(Detector &detector, const vector<string> &fileNames, const string &outFileName)
{
    string fileLine = "";
    for (size_t i = 0; i < fileNames.size(); i++)
    {
        string fileName = fileNames[i];
        Mat img = imread(fileName, cv::IMREAD_COLOR);
        cout << "Processing " << fileName << endl;

        vector<int> labels;
        vector<Rect> rects;
        vector<double> scores;
        TIMER_START(DETECTION);
        detector.DetectMultiScale(img, labels, scores, rects);
        TIMER_END(DETECTION);

        fileLine += fileName + "\n" + to_string(rects.size()) + "\n";
        for (size_t j = 0; j < rects.size(); j++)
        {
            fileLine += to_string(rects[j].x) + " " + to_string(rects[j].y) + " "
                + to_string(rects[j].width) + " " + to_string(rects[j].height) + " "
                + to_string(scores[j]) + " \n";
        }
    }
    ofstream res_file(outFileName);
    res_file << fileLine;
}

#endif


int main(int argc, char** argv)
{
    cv::CommandLineParser args(argc, argv, params);
    if (args.get<bool>("help")) {
        args.printParams();
        return 0;
    }

    ifstream annotationFile(args.get<string>("annotation"));
    FileStorage fs(args.get<string>("config"), FileStorage::READ);

    if (!annotationFile.is_open() || !fs.isOpened())
    {
        cout << "Failed to open input files" << endl;
        return 0;
    }

    vector<string> imagesList;
    string s;
    while (std::getline(annotationFile, s))
    {
        imagesList.push_back(s);
    }

    FileNode detectorParamsNode = fs.root();

    ClassifierFactory factory;
    shared_ptr<Classifier> classifier = factory.CreateClassifier(CAFFE_CLASSIFIER);
    classifier->SetParams(detectorParamsNode);
    classifier->Init();

    int stride, minNeighbours, groupRects, kPyramidLevels;
    Size windowSize, maxWindowSize, minWindowSize;
    string outFileName;
    detectorParamsNode["output_file_name"] >> outFileName;
    detectorParamsNode["step"] >> stride;
    detectorParamsNode["min_neighbs"] >> minNeighbours;
    detectorParamsNode["group_rect"] >> groupRects;
    detectorParamsNode["win_size"] >> windowSize;
    detectorParamsNode["max_win_size"] >> maxWindowSize;
    detectorParamsNode["min_win_size"] >> minWindowSize;
    detectorParamsNode["pyramid_levels_num"] >> kPyramidLevels;

    cout << stride << " " << minNeighbours << " " << groupRects << " "
         << windowSize.width << " " << windowSize.height << " "
         << maxWindowSize.width << " " << maxWindowSize.height << " "
         << minWindowSize.width << " " << minWindowSize.height << " "
         << kPyramidLevels << endl;

    Detector detector(classifier, windowSize, maxWindowSize, minWindowSize,
                      kPyramidLevels, stride, stride, minNeighbours, groupRects);

    TIMER_START(OVERALL);
    detect(detector, imagesList, outFileName);
    TIMER_END(OVERALL);

    return 0;
}
