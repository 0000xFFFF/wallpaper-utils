#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "globals.hpp"
#include "utils.hpp"

enum ALGORITHM {
    KMEANS,
    KMEANSOPT,
    HISTOGRAM
};

enum ACTION { NONE,
              MOVE,
              COPY };

struct ColorInfo {
    cv::Vec3b color;
    double weight;
    double saturation;
    double brightness;
    double hue;
};

struct ImageInfo {
    std::string path;
    std::string filename;
    std::vector<ColorInfo> dominantColors;
    std::string assignedGroup;
    std::string assignedGroupId;
    double groupScore;
};

std::vector<ImageInfo> images;

// Predefined color groups with representative colors (HSV ranges)
struct ColorGroup {
    std::string name;
    float hueMin, hueMax;
    float satMin, satMax;
    float brightMin, brightMax;
    cv::Vec3b representativeColor;
    int counter = 0;
};

std::vector<ColorGroup> colorGroups = {
    {"Miscellaneous", 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, cv::Vec3b(0, 0, 0)},
    {"Blue_Cool", 200, 260, 0.3f, 1.0f, 0.3f, 1.0f, cv::Vec3b(255, 100, 50)},
    {"Red_Warm", 340, 20, 0.3f, 1.0f, 0.3f, 1.0f, cv::Vec3b(50, 50, 255)},
    {"Green_Nature", 80, 140, 0.3f, 1.0f, 0.3f, 1.0f, cv::Vec3b(50, 255, 100)},
    {"Orange_Sunset", 20, 50, 0.4f, 1.0f, 0.4f, 1.0f, cv::Vec3b(50, 165, 255)},
    {"Purple_Mystical", 260, 300, 0.3f, 1.0f, 0.3f, 1.0f, cv::Vec3b(255, 50, 200)},
    {"Yellow_Bright", 50, 80, 0.4f, 1.0f, 0.5f, 1.0f, cv::Vec3b(50, 255, 255)},
    {"Pink_Soft", 300, 340, 0.3f, 1.0f, 0.4f, 1.0f, cv::Vec3b(200, 100, 255)},
    {"Cyan_Tech", 160, 200, 0.4f, 1.0f, 0.4f, 1.0f, cv::Vec3b(255, 200, 100)},
    {"Dark_Moody", 0, 360, 0.0f, 1.0f, 0.0f, 0.25f, cv::Vec3b(40, 40, 40)},
    {"Light_Minimal", 0, 360, 0.0f, 0.3f, 0.8f, 1.0f, cv::Vec3b(240, 240, 240)},
    {"Monochrome", 0, 360, 0.0f, 0.15f, 0.25f, 0.8f, cv::Vec3b(128, 128, 128)},
    {"Earth_Tones", 25, 45, 0.2f, 0.7f, 0.3f, 0.7f, cv::Vec3b(100, 150, 200)}};

std::mutex coutMutex;
std::mutex processMutex;

void calculateColorProperties(ColorInfo& colorInfo)
{
    cv::Mat bgrPixel(1, 1, CV_8UC3, cv::Scalar(colorInfo.color[0], colorInfo.color[1], colorInfo.color[2]));
    cv::Mat hsvPixel;
    cv::cvtColor(bgrPixel, hsvPixel, cv::COLOR_BGR2HSV);

    cv::Vec3b hsv = hsvPixel.at<cv::Vec3b>(0, 0);
    colorInfo.hue = hsv[0] * 2.0;
    colorInfo.saturation = hsv[1] / 255.0;
    colorInfo.brightness = hsv[2] / 255.0;
}

std::vector<ColorInfo> extractDominantColorsHistogram(const cv::Mat& image, int k = 5)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Create histogram
    int hbins = 36, sbins = 16, vbins = 16; // Reasonable resolution
    int histSize[] = {hbins, sbins, vbins};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    float vranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges, vranges};
    int channels[] = {0, 1, 2};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges);

    // Find dominant colors by finding histogram peaks
    std::vector<ColorInfo> colors;
    std::vector<std::tuple<int, int, int, float>> peaks;

    // Extract all non-zero histogram bins
    for (int h = 0; h < hbins; h++) {
        for (int s = 0; s < sbins; s++) {
            for (int v = 0; v < vbins; v++) {
                float count = hist.at<float>(h, s, v);
                if (count > 0) {
                    peaks.emplace_back(h, s, v, count);
                }
            }
        }
    }

    // Sort by count (descending)
    std::sort(peaks.begin(), peaks.end(),
              [](const auto& a, const auto& b) {
                  return std::get<3>(a) > std::get<3>(b);
              });

    // Take top k peaks
    int totalPixels = image.rows * image.cols;
    int numColors = std::min(k, static_cast<int>(peaks.size()));

    for (int i = 0; i < numColors; i++) {
        auto [h_idx, s_idx, v_idx, count] = peaks[i];

        // Convert histogram indices back to HSV values
        float hue = (h_idx + 0.5f) * 180.0f / hbins;
        float sat = (s_idx + 0.5f) * 256.0f / sbins;
        float val = (v_idx + 0.5f) * 256.0f / vbins;

        // Convert HSV to BGR
        cv::Mat hsvPixel(1, 1, CV_32FC3, cv::Scalar(hue, sat, val));
        cv::Mat bgrPixel;
        cv::cvtColor(hsvPixel, bgrPixel, cv::COLOR_HSV2BGR);

        cv::Vec3f bgr = bgrPixel.at<cv::Vec3f>(0, 0);

        ColorInfo colorInfo;
        colorInfo.color = cv::Vec3b(
            static_cast<uchar>(std::clamp(bgr[0], 0.0f, 255.0f)),
            static_cast<uchar>(std::clamp(bgr[1], 0.0f, 255.0f)),
            static_cast<uchar>(std::clamp(bgr[2], 0.0f, 255.0f)));
        colorInfo.weight = count / totalPixels;
        colorInfo.hue = hue * 2.0; // Convert to 0-360 range
        colorInfo.saturation = sat / 255.0;
        colorInfo.brightness = val / 255.0;

        colors.push_back(colorInfo);
    }

    return colors;
}

std::vector<ColorInfo> extractDominantColorsKmeansOpt(const cv::Mat& image, int k = 5)
{
    // Reduce image size for faster processing
    cv::Mat smallImage;
    int maxDim = 150; // Much smaller than 800x600
    if (image.rows > maxDim || image.cols > maxDim) {
        double scale = std::min((double)maxDim / image.rows, (double)maxDim / image.cols);
        cv::resize(image, smallImage, cv::Size(), scale, scale);
    }
    else {
        smallImage = image;
    }

    // Direct conversion to float data without reshaping
    int totalPixels = smallImage.rows * smallImage.cols;
    cv::Mat data(totalPixels, 3, CV_32F);

    // Manually copy pixel data to avoid reshape overhead
    const cv::Vec3b* srcPtr = smallImage.ptr<cv::Vec3b>();
    float* dstPtr = data.ptr<float>();

    for (int i = 0; i < totalPixels; i++) {
        dstPtr[i * 3 + 0] = srcPtr[i][0]; // B
        dstPtr[i * 3 + 1] = srcPtr[i][1]; // G
        dstPtr[i * 3 + 2] = srcPtr[i][2]; // R
    }

    cv::Mat labels, centers;
    cv::kmeans(data, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), // Reduced iterations
               1, cv::KMEANS_PP_CENTERS, centers);                                         // Reduced attempts

    std::vector<int> counts(k, 0);
    for (int i = 0; i < labels.rows; i++) {
        counts[labels.at<int>(i)]++;
    }

    std::vector<ColorInfo> colors;

    for (int i = 0; i < k; i++) {
        ColorInfo colorInfo;
        colorInfo.color = cv::Vec3b(
            static_cast<uchar>(std::clamp(centers.at<float>(i, 0), 0.0f, 255.0f)),
            static_cast<uchar>(std::clamp(centers.at<float>(i, 1), 0.0f, 255.0f)),
            static_cast<uchar>(std::clamp(centers.at<float>(i, 2), 0.0f, 255.0f)));
        colorInfo.weight = (double)counts[i] / totalPixels;
        calculateColorProperties(colorInfo);
        colors.push_back(colorInfo);
    }

    std::sort(colors.begin(), colors.end(),
              [](const ColorInfo& a, const ColorInfo& b) {
                  return a.weight > b.weight;
              });

    return colors;
}
std::vector<ColorInfo> extractDominantColorsKmeans(const cv::Mat& image, int k = 5)
{
    cv::Mat data = image.reshape(1, image.rows * image.cols);
    data.convertTo(data, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(data, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<int> counts(k, 0);
    for (int i = 0; i < labels.rows; i++) {
        counts[labels.at<int>(i)]++;
    }

    std::vector<ColorInfo> colors;
    int totalPixels = image.rows * image.cols;

    for (int i = 0; i < k; i++) {
        ColorInfo colorInfo;
        colorInfo.color = cv::Vec3b(
            static_cast<uchar>(centers.at<float>(i, 0)),
            static_cast<uchar>(centers.at<float>(i, 1)),
            static_cast<uchar>(centers.at<float>(i, 2)));
        colorInfo.weight = (double)counts[i] / totalPixels;
        calculateColorProperties(colorInfo);
        colors.push_back(colorInfo);
    }

    std::sort(colors.begin(), colors.end(),
              [](const ColorInfo& a, const ColorInfo& b) {
                  return a.weight > b.weight;
              });

    return colors;
}

double calculateGroupScore(const std::vector<ColorInfo>& colors, const ColorGroup& group)
{
    double score = 0.0;
    double totalWeight = 0.0;

    for (const auto& color : colors) {
        double colorScore = 0.0;

        // Check hue match (handle wraparound for red)
        bool hueMatch = false;
        if (group.hueMin > group.hueMax) { // wraparound case (red)
            hueMatch = (color.hue >= group.hueMin || color.hue <= group.hueMax);
        }
        else {
            hueMatch = (color.hue >= group.hueMin && color.hue <= group.hueMax);
        }

        if (hueMatch &&
            color.saturation >= group.satMin && color.saturation <= group.satMax &&
            color.brightness >= group.brightMin && color.brightness <= group.brightMax) {
            colorScore = 1.0;
        }
        else {
            // Partial scoring for near matches
            double hueDist = 0.0;
            if (group.hueMin > group.hueMax) {
                hueDist = std::min({std::abs(color.hue - group.hueMin),
                                    std::abs(color.hue - group.hueMax),
                                    std::abs(color.hue - (group.hueMin - 360)),
                                    std::abs(color.hue - (group.hueMax + 360))}) /
                          180.0;
            }
            else {
                hueDist = std::min(std::abs(color.hue - group.hueMin),
                                   std::abs(color.hue - group.hueMax)) /
                          180.0;
            }

            double satDist = std::max(0.0, std::max(group.satMin - color.saturation,
                                                    color.saturation - group.satMax));
            double brightDist = std::max(0.0, std::max(group.brightMin - color.brightness,
                                                       color.brightness - group.brightMax));

            colorScore = std::max(0.0, 1.0 - (hueDist + satDist + brightDist) / 3.0);
        }

        score += colorScore * color.weight;
        totalWeight += color.weight;
    }

    return totalWeight > 0 ? score / totalWeight : 0.0;
}

void assignImageToGroup(ImageInfo& imageInfo)
{
    double bestScore = 0.0;
    int bestGroupId = 0;
    std::string bestGroupName = colorGroups[bestGroupId].name;

    for (size_t i = 1; i < colorGroups.size(); i++) {
        double score = calculateGroupScore(imageInfo.dominantColors, colorGroups[i]);
        if (score > bestScore) {
            bestScore = score;
            bestGroupId = i;
            bestGroupName = colorGroups[i].name;
        }
    }

    imageInfo.assignedGroupId = bestGroupId;
    imageInfo.assignedGroup = bestGroupName;
    imageInfo.groupScore = bestScore;

    // If no group has a good score, assign to miscellaneous
    if (bestScore < 0.3) {
        bestGroupId = 0;
        imageInfo.assignedGroupId = bestGroupId;
        imageInfo.assignedGroup = colorGroups[bestGroupId].name;
    }

    {
        colorGroups[bestGroupId].counter++;
        std::lock_guard<std::mutex> lock(processMutex);
    }
}

size_t scanFolderMakeStructs(const std::string& folderPath)
{
    std::cout << "Scanning folder: " << folderPath << std::endl;

    if (!fs_exists(folderPath)) {
        std::cerr << "Error scanning folder: " << folderPath << std::endl;
        return 0;
    }

    try {
        size_t files = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
            if (entry.is_regular_file() && isSupportedFormat(entry.path().filename().string())) {
                files++;
            }
        }
        images.reserve(files);

        for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
            if (entry.is_regular_file() && isSupportedFormat(entry.path().filename().string())) {
                ImageInfo imgInfo;
                imgInfo.path = entry.path().string();
                imgInfo.filename = entry.path().filename().string();
                images.push_back(imgInfo);
            }
        }
    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Error scanning folder: " << ex.what() << std::endl;
        return 0;
    }

    int totalCount = images.size();
    std::cout << "Found " << totalCount << " image files." << std::endl;

    if (totalCount == 0) {
        std::cout << "No images found." << std::endl;
    }

    return totalCount;
}

void processImages(const std::string& inputFolder, ALGORITHM algorithm)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    size_t count = scanFolderMakeStructs(inputFolder);
    if (!(count > 0)) { exit(1); }

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback in case detection fails

    std::cout << "Using " << numThreads << " threads for processing." << std::endl;

    size_t totalImages = images.size();
    size_t chunkSize = (totalImages + numThreads - 1) / numThreads;
    std::atomic<int> processedImages{0};

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    std::atomic<bool> running = true;

    Cursor::hide();
    Cursor::termClear();

    std::thread printThread([&running, &processedImages, &totalImages]() {
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point prev_time = start_time;
        size_t prev_processed = 0;

        // Moving average for i/s calculation
        std::vector<double> speed_samples;
        const size_t max_samples = 10; // Average over last 10 samples
        float top_speed = 0.0f;

        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            {
                std::lock_guard<std::mutex> lock(coutMutex);
                Cursor::reset();

                for (size_t i = 0; i < colorGroups.size(); i++) {
                    std::cout << colorGroups[i].name << "\t:\t" << colorGroups[i].counter << std::endl;
                }

                size_t current = processedImages;
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_time = now - start_time;
                std::chrono::duration<double> time_delta = now - prev_time;

                // Calculate instantaneous speed
                double instant_speed = 0.0;
                if (time_delta.count() > 0) {
                    instant_speed = (current - prev_processed) / time_delta.count();
                }

                // Add to moving average (only if we processed some images)
                if (current > prev_processed) {
                    speed_samples.push_back(instant_speed);
                    if (speed_samples.size() > max_samples) {
                        speed_samples.erase(speed_samples.begin());
                    }
                }

                // Calculate averaged speed
                double avg_speed = 0.0;
                if (!speed_samples.empty()) {
                    double sum = 0.0;
                    for (double speed : speed_samples) {
                        sum += speed;
                    }
                    avg_speed = sum / speed_samples.size();
                }

                prev_time = now;
                prev_processed = current;

                std::cout << std::endl;
                float p = static_cast<float>(current) / static_cast<float>(totalImages);

                // Calculate ETA
                std::string eta_str = "";
                if (avg_speed > 0 && current < totalImages) {
                    double remaining_time = (totalImages - current) / avg_speed;
                    int eta_minutes = static_cast<int>(remaining_time / 60);
                    int eta_seconds = static_cast<int>(remaining_time) % 60;
                    eta_str = " ETA: " + std::to_string(eta_minutes) + "m " + std::to_string(eta_seconds) + "s";
                }

                if (avg_speed > top_speed) top_speed = avg_speed;

                std::cout << "==: " << current << "/" << totalImages << "  "
                          << std::fixed << std::setprecision(1)
                          << p * 100 << "% (avg: " << std::setprecision(1) << avg_speed << " i/s)" << " (top: " << top_speed << " i/s)"
                          << eta_str << "               " << std::endl;
            }
        }
        std::cout << std::endl;
    });

    auto processImageThread = [&processedImages, &algorithm](size_t start, size_t end, int threadId) {
        for (size_t i = start; i < end; ++i) {
            auto& imageInfo = images[i];

            cv::Mat image = cv::imread(imageInfo.path);
            if (image.empty()) {
                std::lock_guard<std::mutex> lock(coutMutex);
                std::cerr << "[Thread " << threadId << "] Could not load: " << imageInfo.path << std::endl;
                continue;
            }

            if (image.cols > 800 || image.rows > 600) {
                double scale = std::min(800.0 / image.cols, 600.0 / image.rows);
                cv::resize(image, image, cv::Size(), scale, scale);
            }

            switch (algorithm) {
                case KMEANS:    imageInfo.dominantColors = extractDominantColorsKmeans(image); break;
                case KMEANSOPT: imageInfo.dominantColors = extractDominantColorsKmeansOpt(image); break;
                case HISTOGRAM: imageInfo.dominantColors = extractDominantColorsHistogram(image); break;
            }

            assignImageToGroup(imageInfo);
            processedImages++;
        }
    };

    for (int t = 0; t < numThreads; ++t) {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, totalImages);
        if (start >= totalImages) break;
        threads.emplace_back(processImageThread, start, end, t);
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // stop print thread
    running = false;
    printThread.join();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nCompleted in " << duration.count() << "ms" << std::endl;
    std::cout << "Average: " << std::fixed << std::setprecision(2)
              << (double)duration.count() / images.size() << "ms per image" << std::endl;
}

void createGroupFoldersMoveOrCopyFiles(const std::string& outputPath, ACTION action)
{
    try {
        std::filesystem::create_directories(outputPath);

        std::map<std::string, std::vector<ImageInfo*>> groupedImages;

        // Group images by assigned category
        for (auto& image : images) {
            if (!image.assignedGroup.empty()) {
                groupedImages[image.assignedGroup].push_back(&image);
            }
        }

        // Create folders and copy/move images
        for (const auto& group : groupedImages) {
            std::string groupPath = outputPath + "/" + group.first;
            std::filesystem::create_directories(groupPath);

            std::cout << "\n"
                      << group.first << " (" << group.second.size() << " images):" << std::endl;

            for (const auto& image : group.second) {
                std::string destPath = groupPath + "/" + image->filename;

                switch (action) {
                    case NONE: break;
                    case COPY:
                        {
                            try {
                                std::filesystem::copy_file(image->path, destPath, std::filesystem::copy_options::overwrite_existing);
                                std::cout << "  Copied: " << image->filename << " (score: " << std::fixed << std::setprecision(2) << image->groupScore << ")" << std::endl;
                            }
                            catch (const std::filesystem::filesystem_error& ex) {
                                std::cerr << "  Error copying " << image->filename << ": " << ex.what() << std::endl;
                            }

                            break;
                        }
                    case MOVE:
                        {
                            try {
                                std::filesystem::rename(image->path, destPath);
                                std::cout << "  Moved: " << image->filename << " (score: " << std::fixed << std::setprecision(2) << image->groupScore << ")" << std::endl;
                            }
                            catch (const std::filesystem::filesystem_error& ex) {
                                std::cerr << "  Error moving " << image->filename << ": " << ex.what() << std::endl;
                            }

                            break;
                        }
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Error creating output folders: " << ex.what() << std::endl;
    }
}

void generateReport(const std::string& reportPath = "grouping_report.txt")
{
    std::ofstream report(reportPath);

    std::map<std::string, std::vector<ImageInfo*>> groupedImages;
    for (auto& image : images) {
        if (!image.assignedGroup.empty()) {
            groupedImages[image.assignedGroup].push_back(&image);
        }
    }

    report << "WALLPAPER GROUPING REPORT\n";
    report << "=========================\n\n";
    report << "Total images processed: " << images.size() << "\n\n";

    for (const auto& group : groupedImages) {
        report << group.first << " (" << group.second.size() << " images)\n";
        report << std::string(group.first.length() + 20, '-') << "\n";

        for (const auto& image : group.second) {
            report << "  " << image->filename << " (confidence: "
                   << std::fixed << std::setprecision(2) << image->groupScore << ")\n";
        }
        report << "\n";
    }

    report.close();
    std::cout << "Report saved to: " << reportPath << std::endl;
}

void printSummary()
{
    std::map<std::string, int> groupCounts;
    for (const auto& image : images) {
        if (!image.assignedGroup.empty()) {
            groupCounts[image.assignedGroup]++;
        }
    }

    std::cout << "\n=== GROUPING SUMMARY ===" << std::endl;
    std::cout << "Total images: " << images.size() << std::endl;

    for (const auto& group : groupCounts) {
        double percentage = (double)group.second / images.size() * 100;
        std::cout << group.first << ": " << group.second << " images ("
                  << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
    }
}

void handleCtrlC(int)
{
    std::cout << std::endl
              << std::endl;
    Cursor::show();
    exit(1);
}

int main(int argc, char* argv[])
{

    freopen("/dev/null", "w", stderr); // suppress errors

    argparse::ArgumentParser program("grouper", VERSION);
    program.add_description("group wallpapers by color palette");
    auto& options_required = program.add_group("Required");
    options_required.add_argument("-i", "--input")
        .help("input folder")
        .required();
    program.add_argument("-r", "--report")
        .help("save report in a txt file")
        .default_value("")
        .metavar("report.txt");
    auto& options_optional = program.add_group("Optional");
    options_optional.add_argument("-o", "--output")
        .help("output folder (if not speicifed files won't be moved/copied, must specify --copy or --move to do action)");
    auto& mutex_group = options_optional.add_mutually_exclusive_group();
    mutex_group.add_argument("-c", "--copy")
        .help("copy files to output dir")
        .default_value(false)
        .implicit_value(true);
    mutex_group.add_argument("-m", "--move")
        .help("move files to output dir")
        .default_value(false)
        .implicit_value(true);
    options_optional.add_argument("-a", "--algorithm")
        .help("which algorithm to use when grouping images (KMeans = 0, KMeansOptimized = 1, Histogram = 2")
        .metavar("0/1/2")
        .default_value(0)
        .scan<'i', int>();

    // HANDLE CTRL+C
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = handleCtrlC;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    ACTION action = NONE;
    if (program.get<bool>("copy")) { action = COPY; }
    else if (program.get<bool>("move")) {
        action = MOVE;
    }

    ALGORITHM algorithm = KMEANS;
    switch (program.get<int>("algorithm")) {
        case 0: algorithm = KMEANS; break;
        case 1: algorithm = KMEANSOPT; break;
        case 2: algorithm = HISTOGRAM; break;
    }

    std::string inputFolder = program.get<std::string>("input");

    processImages(inputFolder, algorithm);

    // Show summary
    printSummary();

    if (action != NONE) {
        // Create grouped folders
        std::string outputFolder = program.get<std::string>("output");
        createGroupFoldersMoveOrCopyFiles(outputFolder, action);
    }

    std::string reportFile = program.get<std::string>("report");
    if (!reportFile.empty()) {
        generateReport(reportFile);
    }

    std::cout << "\nDone!" << std::endl;
    Cursor::show();

    return 0;
}
