#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "detector.hpp"
#include "classifier_factory.hpp"

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

using namespace std;
using cv::Rect;
using cv::Mat;
using cv::Size;
using cv::FileNode;
using cv::FileStorage;

struct Args {
    string input_path;
    FileNode params_file_node;
    vector<string> filenames;
};

const char* help = "detector \"input/folder\"\n\
    input folder must contain two files:\n\
    annotation.txt - contains filenames of imgs to detect\n\
    config.yml - contains description of classifier:\n\
        step (int)\n\
        min_neighbs (int)\n\
        scale (float)\n\
        group_rect (int 0 or 1)\n\
        device_id - <0 cpu, >=0 gpu\n\
        net_description_file\n\
        net_binary_file\n\
    output of the program will be written in the input folder\n\
        \n";


#if defined(HAVE_MPI) && defined(PAR_SET_IMAGES)

void detect(Detector &detector, vector<string> &fileNames, string &outFileName)
{
    int argc, rank, np, fileStep, leftIdx, rigthIdx;
    char **argv;
    MPI_File file;
    MPI_Status status;
    MPI_Init(&argc, &argv);

    Detector detector(classifier, maxWindowSize, minWindowSize,
                      kPyramidLevels, step, step, min_neighbs, group_rect);

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
        detector.DetectMultiScale(img, labels, scores, rects);
        
        fileLine += fileName + "\n" + to_string(rects.size()) + "\n";
        for (size_t j = 0; j < rects.size(); j++)
        {
            fileLine += to_string(rects[j].x) + " " + to_string(rects[j].y) + " "
                + to_string(rects[j].width) + " " + to_string(rects[j].height) + " "
                + to_string(scores[j]) + " \n";
        }
    }
    MPI_File_open(MPI_COMM_WORLD, (char *)outFileName.c_str(),
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_write_ordered(file, (void *)fileLine.c_str(),
        fileLine.length(), MPI_CHAR, &status);
    MPI_File_close(&file);
    MPI_Finalize();
}

#else

void detect(Detector &detector, string &fileName, string &outFileName)
{    
    Mat img = imread(fileName, cv::IMREAD_COLOR);
    cout << "Processing " << fileName << endl;
    
    vector<int> labels;
    vector<Rect> rects;
    vector<double> scores;

#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
    int argc;
    char **argv;
    MPI_Init(&argc, &argv);
#endif

    detector.DetectMultiScale(img, labels, scores, rects);

    // write to file on 0 process
#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
#endif
    {
        ofstream out(outFileName);
        if (!out.is_open())
        {
            cout << "Error: output file was not opened." << endl;
            cout << help;
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
#if defined(HAVE_MPI) && defined(PAR_PYRAMID)
    MPI_Finalize();
#endif
}

void detect(Detector &detector, vector<string> &fileNames, string &outFileName)
{
    for (size_t i = 0; i < fileNames.size(); i++)
    {
        detect(detector, fileNames[i], outFileName);
    }    
}
#endif


int main(int argc, char** argv)
{
    cout << "Begin main" << endl;
    if (argc < 2) {
        cout << "Too few arguments\n" << help;
        return 0;
    }

    Args args;    
    args.input_path = string(argv[1]);    

    string annot = args.input_path + "/annotation.txt";
    string config = args.input_path + "/config.yml";

    ifstream content_annot(annot);
    FileStorage fs(config, FileStorage::READ);

    if (!content_annot.is_open() || !fs.isOpened())
    {
        cout << "Cannot find or open input files\n";
        cout << help;
        return 0;
    }

    std::string s;
    while (std::getline(content_annot, s))
    {
        args.filenames.push_back(s);
    }

    args.params_file_node = fs.root();

    ClassifierFactory factory;
    shared_ptr<Classifier> classifier = factory.CreateClassifier(CAFFE_CLASSIFIER);
    classifier->SetParams(args.params_file_node);
    classifier->Init();

    FileNode params = args.params_file_node;
    string outFileName;
    params["output_file_name"] >> outFileName;
    outFileName = args.input_path + "/" + outFileName;
    int step = params["step"];
    int min_neighbs = params["min_neighbs"];
    int group_rect = params["group_rect"];
    Size maxWindowSize(params["max_win_width"], params["max_win_height"]),
         minWindowSize(params["min_win_width"], params["min_win_height"]);
    int kPyramidLevels = params["pyramid_levels_num"];
    cout << step << " " << min_neighbs << " " << group_rect << " "
         << maxWindowSize.width << " " << maxWindowSize.height << " "
         << minWindowSize.width << " " << minWindowSize.height << " "
         << kPyramidLevels << endl;
    Detector detector(classifier, maxWindowSize, minWindowSize,
                      kPyramidLevels, step, step, min_neighbs, group_rect);

    detect(detector, args.filenames, outFileName);

    return 1;
}
