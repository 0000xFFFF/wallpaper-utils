#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "debug.hpp"
#include "globals.hpp"
#include "utils.hpp"

struct DarkScoreResult {
    std::string filePath;
    double score;
};

std::vector<DarkScoreResult> results;
std::mutex resultsMutex;

double computeDarkness(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Warning: could not open " << imagePath << std::endl;
        return -1.0;
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar meanVal = cv::mean(gray);
    double avg_brightness = meanVal[0];
    return 1.0 - (avg_brightness / 255.0);
}

void processImages(std::vector<std::string>& images)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // results.reserve(totalCount);

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback in case detection fails

    std::cout << "Using " << numThreads << " threads for processing." << std::endl;

    size_t totalImages = images.size();
    size_t chunkSize = (totalImages + numThreads - 1) / numThreads;
    std::atomic<int> processedImages{0};

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    std::atomic<bool> running = true;

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
                size_t current = processedImages;
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_time = now - start_time;
                std::chrono::duration<double> time_delta = now - prev_time;

                // Calculate instantaneous speed
                double instant_speed = 0.0;
                if (time_delta.count() > 0) {
                    instant_speed = (current - prev_processed) / time_delta.count();
                }

                // Add to moving average (only if we processed some)
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

                Cursor::cr();
                std::cout << "==: " << current << "/" << totalImages << " "
                          << std::fixed << std::setprecision(1)
                          << p * 100 << "% (avg: " << std::setprecision(1) << avg_speed << " i/s)" << " (top: " << top_speed << " i/s)"
                          << eta_str << "               ";
                std::cout.flush();
            }
        }
        std::cout << std::endl;
    });

    auto processImageThread = [&processedImages, &images](size_t start, size_t end, int threadId) {
        UNUSED(threadId);
        for (size_t i = start; i < end; ++i) {
            DarkScoreResult result;
            result.filePath = images[i];
            result.score = computeDarkness(images[i]);
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                results.push_back(result);
            }
            ++processedImages;
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
    std::cout << "Total files processed: " << results.size() << std::endl;
}

int main(int argc, char* argv[])
{
    freopen("/dev/null", "w", stderr); // suppress errors

    argparse::ArgumentParser program("darkscore", VERSION);
    program.add_description("give darkness score for wallpapers");
    program.add_argument("-i", "--input")
        .required()
        .help("Path to a image file or folder containing images (recursive)");
    program.add_argument("-o", "--output")
        .required()
        .help("Path to output CSV file");

    program.add_argument("-s", "-sd", "--sort", "--sortd")
        .default_value(false)
        .implicit_value(true)
        .help("Sort output by darkness score descending order");

    program.add_argument("-sa", "--sorta")
        .default_value(false)
        .implicit_value(true)
        .help("Sort output by darkness score ascending order");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    std::string inputPath = program.get<std::string>("--input");
    std::vector<std::string> images;
    getImages(images, inputPath);
    if (images.empty()) {
        std::cout << "No valid images found." << std::endl;
        return 1;
    }

    processImages(images);

    if (program.get<bool>("--sort") || program.get<bool>("--sortd")) {
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.score > b.score; });
    }

    if (program.get<bool>("--sorta")) {
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.score < b.score; });
    }

    std::string outputPath = program.get<std::string>("--output");
    if (!outputPath.empty()) {
        bool fileExists = std::ifstream(outputPath).good();
        std::ofstream out(outputPath);
        if (!fileExists) {
            out << "image,darkness\n";
        }

        std::cout << std::fixed << std::setprecision(6);
        out << std::fixed << std::setprecision(6);

        for (const auto& result : results) {
            if (result.score >= 0) {
                std::string abs = std::filesystem::canonical(result.filePath);
                std::cout << abs << " => " << result.score << std::endl;
                out << abs << CSV_DELIM << result.score << "\n";
            }
        }

        out.close();
        std::cout << "Results written to " << outputPath << std::endl;
    }

    return 0;
}
