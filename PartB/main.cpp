// This program is a simple template of an C++ program loading and showing image with OpenCV.
// You can ignore this file and write your own program.
// The program takes a image file name as an argument.

#include <stdio.h>
#include <unordered_map>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>


#define PI 3.141592653589793238463
#define LIN_LEN 200.0
using namespace std;

typedef unordered_map<unsigned long, bool> THashMap;

unsigned long coords_to_hash(int a, int b) {
    assert(a >= 0 && b >= 0);
    // Szudzik's function
    return a >= b ? a * a + a + b : a + b * b; // where a, b >= 0

}

double get_central_moments(vector<tuple<int, int>> *max_area, cv::Point2f *centroid, int k, int j) {
    assert(k >= 0 && j >= 0);
    double moment = 0.0;
    for (auto &i : *max_area) {
        int x = get<1>(i);
        int y = get<0>(i);
        moment += pow(x - centroid->x, k) * pow(y - centroid->y, j);
    }
    return moment / ((double) (k + j + 2.0) / 2.0);
}

double get_pa_angle(vector<tuple<int, int>> *max_area, cv::Point2f *centroid) {
    double pa_11 = get_central_moments(max_area, centroid, 1, 1);
    double pa_20 = get_central_moments(max_area, centroid, 2, 0);
    double pa_02 = get_central_moments(max_area, centroid, 0, 2);
    // this does not conform with the lecture but gives the best results :D
    return 0.5 * atan2(2 * pa_11, pa_02 - pa_20);
}

cv::Point2f *get_centoid(vector<tuple<int, int>> *max_area) {
    double centr_x = 0.0;
    double centr_y = 0.0;
    for (auto &i : *max_area) {
        int x = get<1>(i);
        int y = get<0>(i);
        centr_x += x;
        centr_y += y;
        //colorImg.at<cv::Vec3f>(x, y) = cv::Vec3f(0.0, 120.0, 0);
    }
    centr_x = centr_x / max_area->size();
    centr_y = centr_y / max_area->size();
    return new cv::Point2f(centr_x, centr_y);
}


THashMap *fill_map(int rows, int cols) {
    auto *vc = new THashMap();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            (*vc)[coords_to_hash(i, j)] = true;
        }
    }
    assert(vc->size() == (unsigned int) (rows * cols));
    return vc;
}


vector<tuple<int, int>> *discover_area(cv::Mat image, tuple<int, int> seed, THashMap *vc) {
    queue<tuple<int, int>> img_area;
    auto *out_area = new vector<tuple<int, int>>; // init on heap
    img_area.push(seed);
    while (!img_area.empty()) {
        auto tmp = img_area.front();
        int i = get<0>(tmp);
        int j = get<1>(tmp);
        img_area.pop();
        if (i < image.rows && j < image.cols
            && vc->at(coords_to_hash(i, j))
            && image.at<uchar>(i, j) == 255) {
            out_area->push_back(tmp);
            (*vc)[coords_to_hash(i, j)] = false;
            // insert all neighbors of tmp
            img_area.push(make_tuple(i + 1, j));
            img_area.push(make_tuple(i + 1, j + 1));
            img_area.push(make_tuple(i, j + 1));
            img_area.push(make_tuple(i - 1, j + 1));
            img_area.push(make_tuple(i - 1, j));
            img_area.push(make_tuple(i - 1, j - 1));
            img_area.push(make_tuple(i, j - 1));
            img_area.push(make_tuple(i + 1, j - 1));
        }

    }
    return out_area;
}


vector<vector<tuple<int, int>>> *get_areas(cv::Mat image) {
    auto *max_area = new vector<vector<tuple<int, int>>>;
    auto *vc = fill_map(image.rows, image.cols);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int intensity = image.at<uchar>(i, j);
            const unsigned long key = coords_to_hash(i, j);
            if (intensity != 0) {
                if (vc->at(key)) {
                    vector<tuple<int, int>> *area = discover_area(image, make_tuple(i, j), vc);
                    if (area->size() > 100) { // everything below 100 px is considered as noise
                        max_area->push_back(*area);
                    }
                }
            } else {
                if (vc->at(key)) {
                    (*vc)[key] = false;
                }

            }


        }
    }
    return max_area;
}

int main(int argc, char **argv) {
    int c;
    string path;
    cv::Mat workImage, srcImage, colorImg;
    vector<vector<tuple<int, int>>> *areas;

    while ((c = getopt(argc, argv, "p:h")) != -1)
        switch (c) {
            case 'p':
                path = optarg;
                break;
            case 'h':
                cout << "*** Binary Maschine Vision ***" << endl;
                cout << "pass the path to your image with -p <path>" << endl;
                return 1;
            default:
                perror("No path: use -h to get help");
        }

    if (path.empty()) {
        cerr << "Path not provided";
        return 1;
    }

    cout << "Performing analysis for image: " << path << endl;
    // Load a gray scale picture.
    srcImage = cv::imread(path, 0);
    if (!srcImage.data) {
        perror("Could not load image:");
        exit(1);
    }
    cv::cvtColor(srcImage, colorImg, cv::COLOR_GRAY2RGB);

    // Create windows for debug.
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);


    // Show the source image.
    //cv::imshow("SrcImage", SrcImage);
    //cv::waitKey();

    // Duplicate the source iamge.
    workImage = srcImage.clone();


    //Extract the contour of
    cv::GaussianBlur(workImage, workImage, cv::Size(3, 3), 0, 0);
    cv::threshold(workImage, workImage, 128, 255, cv::THRESH_BINARY);


    // Opening
    cv::erode(workImage, workImage, cv::Mat());
    cv::dilate(workImage, workImage, cv::Mat());

    areas = get_areas(workImage);
    for (auto max_area: *areas) {
        auto centroid = get_centoid(&max_area);
        double pa_angle_1 = get_pa_angle(&max_area, centroid);
        double x_1 = centroid->x + LIN_LEN * sin(pa_angle_1);
        double y_1 = centroid->y + LIN_LEN * cos(pa_angle_1);
        double x_0 = centroid->x - LIN_LEN * sin(pa_angle_1);
        double y_0 = centroid->y - LIN_LEN * cos(pa_angle_1);
        cv::circle(colorImg, *centroid, 5, cv::Scalar(0, 0, 255.0), 1);
        cv::circle(colorImg, *centroid, 1, cv::Scalar(0, 0, 255.0), 1);
        cv::line(colorImg, cv::Point2f(x_0, y_0), cv::Point2f(x_1, y_1), cv::Scalar(0, 0, 255.0), 1);
        cout << "x: " << centroid->x << " "
             << "y: " << centroid->y << " "
             << "p_a: " << pa_angle_1 * (180.0 / PI) << endl;
    }

    cv::imshow("Output", colorImg);
    cv::waitKey();
    return 0;
}
