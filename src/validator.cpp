#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "globals.hpp"
#include "utils.hpp"
#include "debug.hpp"

struct ValidationResult {
    std::string filePath;
    std::string filename;
    bool isValid;
    int width;
    int height;
};

std::vector<ValidationResult> results;
std::mutex resultsMutex;

std::atomic<int> corruptedCount = 0;

ValidationResult validateImage(const std::string& imagePath)
{
    ValidationResult result;
    result.filePath = imagePath;
    result.filename = std::filesystem::path(imagePath).filename().string();
    result.isValid = false;
    result.width = 0;
    result.height = 0;

    try {
        cv::Mat image = cv::imread(imagePath);
        if (!image.empty()) {
            result.isValid = true;
            result.width = image.cols;
            result.height = image.rows;
        }
    }
    catch (const cv::Exception& e) {
        // OpenCV exception - image is corrupted
        result.isValid = false;
    }
    catch (...) {
        // Any other exception
        result.isValid = false;
    }

    if (!result.isValid) { corruptedCount++; }

    return result;
}

void processImages(std::vector<std::string>& images)
{
    auto startTime = std::chrono::high_resolution_clock::now();

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
                size_t corrupted = corruptedCount;
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
                std::cout << "==: " << current << "/" << totalImages << " (bad: " << corrupted << ") "
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
            ValidationResult result = validateImage(images[i]);
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
    std::cout << "Valid images: " << (results.size() - corruptedCount) << std::endl;
    std::cout << "Corrupted/unreadable images: " << corruptedCount << std::endl;

    if (corruptedCount > 0) {
        std::cout << "\nCorrupted files:" << std::endl;
        for (const auto& result : results) {
            if (!result.isValid) {
                std::cout << "  " << result.filePath << std::endl;
            }
        }
    }
}

void deleteCorruptedFiles()
{
    std::vector<std::string> corruptedFiles;
    for (const auto& result : results) {
        if (!result.isValid) {
            corruptedFiles.push_back(result.filePath);
        }
    }

    if (corruptedFiles.empty()) {
        std::cout << "No corrupted files to delete." << std::endl;
        return;
    }

    char response;
    std::cout << "\nDo you want to DELETE all " << corruptedFiles.size()
              << " corrupted files? (y/N): ";
    std::cin >> response;

    if (response == 'y' || response == 'Y') {
        int deletedCount = 0;
        int errorCount = 0;
        for (const auto& file : corruptedFiles) {
            try {
                if (std::filesystem::remove(file)) {
                    std::cout << "Deleted: " << file << std::endl;
                    deletedCount++;
                }
                else {
                    std::cout << "Failed to delete: " << file << std::endl;
                }
            }
            catch (const std::filesystem::filesystem_error& ex) {
                std::cout << "Error deleting " << file << ": " << ex.what() << std::endl;
                errorCount++;
            }
        }
        std::cout << "\nDeleted " << deletedCount << " corrupted files." << std::endl;
        std::cout << "\nErrors: " << errorCount << std::endl;
    }
    else {
        std::cout << "Deletion cancelled." << std::endl;
    }
}

void moveCorruptedFiles(const std::string& quarantineFolder = "corrupted_images")
{
    std::vector<std::string> corruptedFiles;
    for (const auto& result : results) {
        if (!result.isValid) {
            corruptedFiles.push_back(result.filePath);
        }
    }

    if (corruptedFiles.empty()) {
        std::cout << "No corrupted files to move." << std::endl;
        return;
    }

    try {
        std::filesystem::create_directories(quarantineFolder);

        int movedCount = 0;
        for (const auto& file : corruptedFiles) {
            std::filesystem::path sourcePath(file);
            std::filesystem::path destPath = std::filesystem::path(quarantineFolder) / sourcePath.filename();

            // Handle filename conflicts
            int counter = 1;
            while (std::filesystem::exists(destPath)) {
                std::string stem = sourcePath.stem().string();
                std::string extension = sourcePath.extension().string();
                destPath = std::filesystem::path(quarantineFolder) / (stem + "_" + std::to_string(counter) + extension);
                counter++;
            }

            try {
                std::filesystem::rename(file, destPath);
                std::cout << "Moved: " << sourcePath.filename() << " -> " << destPath << std::endl;
                movedCount++;
            }
            catch (const std::filesystem::filesystem_error& ex) {
                std::cout << "Error moving " << file << ": " << ex.what() << std::endl;
            }
        }
        std::cout << "\nMoved " << movedCount << " corrupted files to '" << quarantineFolder << "' folder." << std::endl;
    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cout << "Error creating quarantine folder: " << ex.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    freopen("/dev/null", "w", stderr); // suppress errors

    argparse::ArgumentParser program("validator", VERSION);
    program.add_description("validate images, find corrupt images (and delete them/move them/etc)");
    program.add_argument("-i", "--input")
        .required()
        .help("Path to a image file or folder containing images (recursive)");

    program.add_argument("-m", "--move")
        .default_value(false)
        .implicit_value(true)
        .help("move corrupt files to corrupted_images folder (make one)");

    program.add_argument("-d", "--delete")
        .default_value(false)
        .implicit_value(true)
        .help("delete corrupt files");

    program.add_argument("-p", "--prompt")
        .default_value(false)
        .implicit_value(true)
        .help("prompt what to do after scanning (nothing/delete/move)");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    int choice = 0;

    if      (program.get<bool>("delete")) { choice = 1; }
    else if (program.get<bool>("move"))   { choice = 2; }


    std::string inputPath = program.get<std::string>("input");
    std::vector<std::string> images;
    getImages(images, inputPath);
    if (images.empty()) {
        std::cout << "No valid images found." << std::endl;
        return 1;
    }
    results.reserve(images.size());

    processImages(images);

    if (program.get<bool>("prompt")) {
        std::cout << "\nWhat would you like to do with corrupted files?" << std::endl;
        std::cout << "0. Do nothing" << std::endl;
        std::cout << "1. Delete them permanently" << std::endl;
        std::cout << "2. Move them to 'corrupted_images' folder" << std::endl;
        std::cout << "Choice (0/1/2): ";

        std::cin >> choice;
    }

    if (corruptedCount > 0) {
        switch (choice) {
            case 0:
                std::cout << "No action taken." << std::endl;
                break;
            case 1:
                deleteCorruptedFiles();
                break;
            case 2:
                moveCorruptedFiles();
                break;
        }
    }

    return 0;
}
