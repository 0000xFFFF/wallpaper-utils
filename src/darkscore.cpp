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
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
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

// Load existing results from CSV
std::unordered_map<std::string, double> loadExistingResults(const std::string& csvPath)
{
    std::unordered_map<std::string, double> cache;
    std::ifstream inFile(csvPath);

    if (!inFile.is_open()) {
        return cache;
    }

    std::string line;
    // Skip header
    std::getline(inFile, line);

    while (std::getline(inFile, line)) {
        if (line.empty()) continue;

        size_t delimPos = line.find(CSV_DELIM);
        if (delimPos != std::string::npos) {
            std::string imagePath = line.substr(0, delimPos);
            std::string scoreStr = line.substr(delimPos + 1);

            try {
                double score = std::stod(scoreStr);
                cache[imagePath] = score;
            }
            catch (const std::exception& e) {
                // Skip invalid lines
                continue;
            }
        }
    }

    inFile.close();
    return cache;
}

void processImages(std::vector<std::string>& images)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

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

        std::vector<double> speed_samples;
        const size_t max_samples = 10;
        float top_speed = 0.0f;

        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            {
                size_t current = processedImages;
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_time = now - start_time;
                std::chrono::duration<double> time_delta = now - prev_time;

                double instant_speed = 0.0;
                if (time_delta.count() > 0) {
                    instant_speed = (current - prev_processed) / time_delta.count();
                }

                if (current > prev_processed) {
                    speed_samples.push_back(instant_speed);
                    if (speed_samples.size() > max_samples) {
                        speed_samples.erase(speed_samples.begin());
                    }
                }

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
    freopen("/dev/null", "w", stderr);

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
    std::string outputPath = program.get<std::string>("--output");

    // Load existing results from CSV
    std::unordered_map<std::string, double> cachedResults = loadExistingResults(outputPath);

    int cached_count = 0;
    int removed_count = 0;
    int new_count = 0;

    if (!cachedResults.empty()) {
        std::cout << "Loaded " << cachedResults.size() << " cached results from " << outputPath << std::endl;

        // Remove entries for files that no longer exist
        for (auto it = cachedResults.begin(); it != cachedResults.end();) {
            if (!std::filesystem::exists(it->first)) {
                it = cachedResults.erase(it);
                removed_count++;
            }
            else {
                ++it;
            }
        }

        if (removed_count > 0) {
            std::cout << "Removed " << removed_count << " entries for non-existent files" << std::endl;
        }
    }

    // Get all images from input
    std::vector<std::string> allImages;
    getImages(allImages, inputPath);
    if (allImages.empty()) {
        std::cout << "No valid images found." << std::endl;
        return 1;
    }

    // Separate images into cached and new
    std::vector<std::string> imagesToProcess;
    for (const auto& imgPath : allImages) {
        std::string absPath = std::filesystem::canonical(imgPath);

        if (cachedResults.find(absPath) != cachedResults.end()) {
            // Already cached, add to results
            DarkScoreResult result;
            result.filePath = absPath;
            result.score = cachedResults[absPath];
            results.push_back(result);
            cached_count++;
        }
        else {
            // Need to process
            imagesToProcess.push_back(imgPath);
            new_count++;
        }
    }

    std::cout << "Images summary:" << std::endl;
    std::cout << "  Cached: " << cached_count << std::endl;
    std::cout << "  New to process: " << new_count << std::endl;
    std::cout << "  Removed (deleted files): " << removed_count << std::endl;
    std::cout << "  Total: " << (cached_count + new_count) << std::endl;

    // Process only new images
    if (!imagesToProcess.empty()) {
        std::cout << "\nProcessing " << imagesToProcess.size() << " new images..." << std::endl;
        processImages(imagesToProcess);
    }
    else {
        std::cout << "\nNo new images to process!" << std::endl;
    }

    // Sort if requested
    if (program.get<bool>("--sort") || program.get<bool>("--sortd")) {
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.score > b.score; });
    }

    if (program.get<bool>("--sorta")) {
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.score < b.score; });
    }

    // Write results
    if (!outputPath.empty()) {
        std::ofstream out(outputPath);
        out << "image,darkness\n";

        std::cout << std::fixed << std::setprecision(6);
        out << std::fixed << std::setprecision(6);

        for (const auto& result : results) {
            if (result.score >= 0) {
                std::cout << result.filePath << " => " << result.score << std::endl;
                out << result.filePath << CSV_DELIM << result.score << "\n";
            }
        }

        out.close();
        std::cout << "\nResults written to " << outputPath << std::endl;
        std::cout << "Final summary:" << std::endl;
        std::cout << "  Total entries in CSV: " << results.size() << std::endl;
        std::cout << "  New entries added: " << new_count << std::endl;
        std::cout << "  Entries removed: " << removed_count << std::endl;
    }

    return 0;
}
