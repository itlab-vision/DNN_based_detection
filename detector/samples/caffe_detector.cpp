#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Detector.hpp"
#include "caffe_classifier.hpp"

using std::vector;
using std::ifstream;
using std::ofstream;
using std::string;
using std::shared_ptr;
using cv::Rect;
using cv::Mat;
using cv::Size;

struct Args {
    char* input;
    char* output;
    int step;
    int min_neighbs;
    float scale;
    int group_rect;
    vector<char*> filenames;
    string net_config;
};

bool parse_args(int argc, char** argv, Args& args);
void detect(Classifier* classifier, Args args, FILE* out);

const char* help = "detector -i \"input/folder\" -o \"output/folder\"\
[--step (int), --min_neighbs (int), --scale (float), \
--group_rect (int 0 or 1)]\n\
    input folder must contain two files:\n\
    annotation.txt - contains filenames of imgs to detect\n\
    net_config.yml - contains description of this type:\n\
        device_id - <0 cpu >=0 gpu\n\
        net_description_file\n\
        net_binary_file\n\
        \n";

bool parse_args(int argc, char **argv, Args& args) {
    if (argc < 3) {
        printf("Too few arguments\n");
        printf("%s", help);
        return false;
    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0) {
                args.input = argv[++i];
            } else if (strcmp(argv[i], "-o") == 0) {
                args.output = argv[++i];
            } else if (strcmp(argv[i], "--step") == 0) {
                args.step = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--min_neighbs") == 0) {
                args.min_neighbs = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--scale") == 0) {
                args.scale = atof(argv[++i]);
            } else if (strcmp(argv[i], "--group_rect") == 0) {
                args.group_rect = atoi(argv[++i]);
            } else {
                printf("Unrecognized param\n");
                printf("%s", help);
                return false;
            }
        }
    }
    return true;
}

void detect(shared_ptr<Classifier> classifier, Args args, FILE* out) {
    vector<int> labels;
    vector<Rect> rects;
    vector<double> scores;

    Detector detector(classifier, Size(227, 227),
                      args.step, args.step, args.scale,
                      args.min_neighbs, args.group_rect);

    for (size_t i = 0; i < args.filenames.size(); i++) {
        Mat img = imread(args.filenames[i], cv::IMREAD_COLOR);
        printf("Processing %s", args.filenames[i]);

        detector.Detect(img, labels, scores, rects);

        fprintf(out, "%s%lu\n\n", args.filenames[i], rects.size());
        for (size_t j = 0; j < rects.size(); j++) {
            fprintf(out, "%u %u %u %u %lf", rects[j].x, rects[j].y,
                rects[j].width, rects[j].height, scores[j]);
        }
        labels.clear();
        scores.clear();
        rects.clear();
    }
}

int main(int argc, char** argv) {
    Args args;
    args.step = 1;
    args.min_neighbs = 3;
    args.scale = 1.2f;
    args.group_rect = 0;

    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    FILE *ant = NULL, *net = NULL;
    char *tmp = new char[strlen(args.input) + 1 + 15];
    strcpy(tmp, args.input);
    ant = fopen(strcat(tmp, "/annotation.txt"), "r");
    strcpy(tmp, args.input);
    net = fopen(strcat(tmp, "/net_config.yml"), "r");
    if (ant == NULL || net == NULL) {
        printf("Cannot find or open input files\n");
        printf("%s", help);
        return 1;
    }

    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, ant)) != -1) {
        char *new_line = new char[strlen(line) + 1];
        strcpy(new_line, line);
        args.filenames.push_back(new_line);
    }
    while ((read = getline(&line, &len, net)) != -1) {
       args.net_config += line;
    }

    shared_ptr<Classifier> classifier(new CaffeClassifier());
    classifier->SetParams(args.net_config);
    classifier->Init();

    FILE* out;
    out = fopen(strcat(args.output, "/result.txt"), "w");
    if (out == NULL) {
        printf("Problems with creating output file\n");
        printf("%s", help);
        return 1;
    }

    detect(classifier, args, out);

    printf("sdfgdsgsdg\n");
    fclose(out);
    return 0;
}
