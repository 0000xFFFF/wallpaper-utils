#include "globals.hpp"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "utils.hpp"

// Global flag to interrupt sleep
std::atomic<bool> g_running{true};
std::atomic<bool> g_sleeping{false};
std::mutex g_sleep_mutex;
std::condition_variable g_sleep_cv;

void handle_signal(int sig)
{
    if (sig == SIGRTMIN + 10) {
        std::cout << "Received SIGRTMIN+10! Triggering wallpaper change...\n";
        g_sleeping = false;
        g_sleep_cv.notify_all();
    }
}

void daemonize()
{
    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "Fork failed\n";
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        // Parent exits
        exit(EXIT_SUCCESS);
    }

    // Child continues
    if (setsid() < 0) {
        exit(EXIT_FAILURE);
    }

    // Catch, ignore, or handle signals here if needed
    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    // Fork again to prevent reacquiring a terminal
    pid = fork();
    if (pid < 0) exit(EXIT_FAILURE);
    if (pid > 0) exit(EXIT_SUCCESS);

    // Set new file permissions
    umask(0);

    // Change working directory to root
    chdir("/");

    // Close stdin
    close(STDIN_FILENO);

    // Redirect stdout and stderr to log file using file descriptors
    int logfd = open("/tmp/darkscore-select.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (logfd != -1) {
        dup2(logfd, STDOUT_FILENO);
        dup2(logfd, STDERR_FILENO);
        if (logfd > 2) {
            close(logfd);
        }
    }
}

// Map darkness score (0=bright, 1=dark)
// bucket 0-5 (0=darkest, 5=brightest)
int getDarknessBucket(double score)
{
    if (score > 0.9) return 0; // very dark
    if (score > 0.8) return 1; // dark
    if (score > 0.6) return 2; // mid-dark
    if (score > 0.4) return 3; // mid-bright
    if (score > 0.2) return 4; // bright
    if (score > 0.0) return 5; // very bright
    return 5;
}

int getTargetBucketForHour(int hour)
{
    if (hour >= 21) return 0; // very dark
    if (hour >= 20) return 1; // dark
    if (hour >= 19) return 2; // mid-dark
    if (hour >= 18) return 3; // mid-bright
    if (hour >= 17) return 4; // bright
    if (hour >= 12) return 5; // very bright
    if (hour >= 9) return 2;  // bright
    if (hour >= 7) return 2;  // mid-dark
    if (hour >= 5) return 1;  // dark
    if (hour >= 0) return 0;  // very dark
    return 0;
}

struct DarkScoreResult {
    std::string filePath;
    double score;
};

std::vector<std::vector<DarkScoreResult>> loadBuckets(const std::string& inputPath)
{
    std::vector<std::vector<DarkScoreResult>> buckets(6);

    std::ifstream file(inputPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << inputPath << std::endl;
        return buckets;
    }

    std::string line;
    if (getline(file, line)) {} // skip header

    while (getline(file, line)) {
        std::vector<std::string> fields = csv_split(line, CSV_DELIM);
        if (fields.size() < 2) continue;

        DarkScoreResult image;
        image.filePath = fields[0];
        try {
            image.score = std::stod(fields[1]);
        }
        catch (...) {
            continue;
        }

        int b = getDarknessBucket(image.score);
        buckets[b].push_back(image);
    }

    return buckets;
}

// State tracker for sequential iteration through buckets
struct BucketIterator {
    std::vector<std::vector<DarkScoreResult>> shuffledBuckets;
    std::vector<size_t> currentIndices; // Current position in each bucket
    int lastUsedBucket;
    std::mt19937 rng;

    BucketIterator(const std::vector<std::vector<DarkScoreResult>>& buckets)
        : shuffledBuckets(buckets), currentIndices(6, 0), lastUsedBucket(-1)
    {
        std::random_device rd;
        rng.seed(rd());

        // Shuffle all buckets initially
        for (auto& bucket : shuffledBuckets) {
            std::shuffle(bucket.begin(), bucket.end(), rng);
        }
    }

    DarkScoreResult getNext(int targetBucket)
    {
        // Find the actual bucket to use (with fallback logic)
        int chosenBucket = targetBucket;
        int offset = 0;
        while (shuffledBuckets[chosenBucket].empty() && offset < 6) {
            offset++;
            int up = targetBucket + offset;
            int down = targetBucket - offset;
            if (up < 6 && !shuffledBuckets[up].empty()) {
                chosenBucket = up;
                break;
            }
            if (down >= 0 && !shuffledBuckets[down].empty()) {
                chosenBucket = down;
                break;
            }
        }

        if (shuffledBuckets[chosenBucket].empty()) {
            throw std::runtime_error("No wallpapers available in any brightness bucket!");
        }

        // If bucket changed, reset and reshuffle the new bucket
        if (chosenBucket != lastUsedBucket) {
            std::cout << "Bucket changed from " << lastUsedBucket
                      << " to " << chosenBucket
                      << ", reshuffling..." << std::endl;
            currentIndices[chosenBucket] = 0;
            std::shuffle(shuffledBuckets[chosenBucket].begin(),
                         shuffledBuckets[chosenBucket].end(),
                         rng);
            lastUsedBucket = chosenBucket;
        }

        // Get current wallpaper from bucket
        size_t& currentIdx = currentIndices[chosenBucket];
        const auto& result = shuffledBuckets[chosenBucket][currentIdx];

        // Advance index, wrap around and reshuffle if we've gone through all
        currentIdx++;
        if (currentIdx >= shuffledBuckets[chosenBucket].size()) {
            std::cout << "Reached end of bucket " << chosenBucket
                      << ", reshuffling..." << std::endl;
            currentIdx = 0;
            std::shuffle(shuffledBuckets[chosenBucket].begin(),
                         shuffledBuckets[chosenBucket].end(),
                         rng);
        }

        return result;
    }
};

void printBucketInfo(const std::vector<std::vector<DarkScoreResult>>& buckets)
{
    std::cout << "Map darkness score (0=bright, 1=dark) → bucket 0-5 (0=darkest, 5=brightest)" << std::endl;
    for (size_t i = 0; i < buckets.size(); i++) {
        std::cout << "bucket " << i << " has " << buckets[i].size() << " images" << std::endl;
    }
}

void executeWallpaperChange(const std::string& execStr, const DarkScoreResult& chosen, int hour, int bucket)
{
    std::time_t now = std::time(nullptr);
    std::cout << "[" << trim(std::string(std::ctime(&now))) << "] ";
    std::cout << "Hour: " << hour
              << " | Bucket: " << bucket
              << " | Selected: " << chosen.filePath
              << " | Score: " << chosen.score << std::endl;

    if (!execStr.empty()) {
        executeCommand(execStr, chosen.filePath);
    }
}

// Check if user pressed a key
bool checkKeyPress()
{
    char c;
    int n = read(STDIN_FILENO, &c, 1);
    return n > 0;
}

// Sleep with ability to skip on keypress or signal
void interruptibleSleep(int sleepMs, bool checkKeys = true)
{
    if (checkKeys) {
        std::cout << "Sleeping for " << (sleepMs / 1000) << "s (press any key or send signal to skip)..." << std::endl;
    }

    g_sleeping = true;
    {
        std::unique_lock<std::mutex> lock(g_sleep_mutex);
        g_sleep_cv.wait_for(lock, std::chrono::milliseconds(sleepMs), [&] { return !g_sleeping; });
    }
}

constexpr int LOOP_SLEEP_MS = 1000 * 60 * 1; // 1 min

int main(int argc, char* argv[])
{
    argparse::ArgumentParser program("darkscore-select", VERSION);
    program.add_description(R"(select wallpaper from csv file based on time of day and darkness score

    (night time = dark wallpaper, day time = bright wallper)

    wallpapers are shuffled into 6 buckets:

    buckets(6):
        darkness score > 0.9    very dark
        darkness score > 0.8    dark
        darkness score > 0.6    mid-dark
        darkness score > 0.4    mid-bright
        darkness score > 0.2    bright
        darkness score > 0.0    very bright

    bucket is chosen by current hour:
        (hour >= 21)    =>    very dark
        (hour >= 20)    =>    dark
        (hour >= 19)    =>    mid-dark
        (hour >= 18)    =>    mid-bright
        (hour >= 17)    =>    bright
        (hour >= 12)    =>    very bright
        (hour >=  9)    =>    bright
        (hour >=  7)    =>    mid-dark
        (hour >=  5)    =>    dark
        (hour >=  0)    =>    very dark)

     wallpapers get reshuffled:
       * after looping through the entire bucket
       * if chosen bucket changes (hour changes)

    notes:
        * You can change wallpaper on enter
        * or by sending a signal (useful when running as a daemon (-d)) with:
        pkill -RTMIN+10 -f wpu-darkscore-select)");

    program.add_argument("-i", "--input")
        .required()
        .help("csv file that was made by bgcpl-darkscore")
        .metavar("file.csv");

    program.add_argument("-e", "--exec")
        .help("pass image to a command and execute (e.g. plasma-apply-wallpaperimage)")
        .metavar("command")
        .default_value("");

    program.add_argument("-d", "--daemon")
        .default_value(false)
        .implicit_value(true)
        .help("run daemon in the background");

    program.add_argument("-l", "--loop")
        .default_value(false)
        .implicit_value(true)
        .help("loop logic for setting wallpapers");

    program.add_argument("-s", "--sleep")
        .help("sleep ms for loop")
        .metavar("sleep_ms")
        .default_value(LOOP_SLEEP_MS)
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    std::string inputPath = program.get<std::string>("input");
    std::string execStr = program.get<std::string>("exec");
    bool isDaemon = program.get<bool>("daemon");
    bool isLoop = program.get<bool>("loop");
    int sleepMs = program.get<int>("sleep");

    try {
        inputPath = std::filesystem::canonical(inputPath).string();
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: Could not resolve path: " << inputPath << " - " << e.what() << std::endl;
        return 1;
    }

    // Register handler for SIGRTMIN+10
    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGRTMIN + 10, &sa, nullptr);
    std::cout << "Running. PID: " << getpid() << "\n";
    std::cout << "Send signal with: pkill -RTMIN+10 -f darkscore-select\n";

    // Daemonize if requested
    if (isDaemon) {
        daemonize();
    }

    // Load buckets once
    auto buckets = loadBuckets(inputPath);

    // Check if any buckets have images
    bool hasImages = false;
    for (const auto& bucket : buckets) {
        if (!bucket.empty()) {
            hasImages = true;
            break;
        }
    }

    if (!hasImages) {
        std::cerr << "Error: No valid images found in CSV file!" << std::endl;
        return 1;
    }

    if (!isDaemon) {
        printBucketInfo(buckets);
    }

    // Main execution logic
    if (isLoop || isDaemon) {
        // Create bucket iterator for sequential iteration
        BucketIterator iterator(buckets);

        std::thread logicThread([&]() {
            while (g_running) {
                try {
                    std::time_t now = std::time(nullptr);
                    std::tm* local = std::localtime(&now);
                    int hour = local->tm_hour;
                    int targetBucket = getTargetBucketForHour(hour);

                    auto chosen = iterator.getNext(targetBucket);
                    executeWallpaperChange(execStr, chosen, hour, targetBucket);

                    // Sleep with interruption support
                    interruptibleSleep(sleepMs, isLoop && !isDaemon);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error in loop: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait before retrying
                }
            }
        });

        if (isDaemon) {
            logicThread.join();
        }
        else {
            while (g_running) {
                char c;
                if (checkKeyPress(&c)) {
                    g_sleeping = false;
                    g_sleep_cv.notify_all();
                }

                if (c == 'q') { break; }
            }
        }

        g_running = false;
        g_sleeping = false;
        g_sleep_cv.notify_all();
        logicThread.join();
    }
    else {
        // Single execution mode - just pick randomly for one-time use
        std::time_t now = std::time(nullptr);
        std::tm* local = std::localtime(&now);
        int hour = local->tm_hour;

        try {
            int targetBucket = getTargetBucketForHour(hour);

            // Find the actual bucket to use (with fallback logic)
            int chosenBucket = targetBucket;
            int offset = 0;
            while (buckets[chosenBucket].empty() && offset < 6) {
                offset++;
                int up = targetBucket + offset;
                int down = targetBucket - offset;
                if (up < 6 && !buckets[up].empty()) {
                    chosenBucket = up;
                    break;
                }
                if (down >= 0 && !buckets[down].empty()) {
                    chosenBucket = down;
                    break;
                }
            }

            if (buckets[chosenBucket].empty()) {
                throw std::runtime_error("No wallpapers available in any brightness bucket!");
            }

            // Random selection for single execution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, static_cast<int>(buckets[chosenBucket].size()) - 1);
            const auto& chosen = buckets[chosenBucket][dist(gen)];

            std::cout << "Current hour: " << hour << std::endl;
            std::cout << "Target bucket: " << targetBucket << " (used " << chosenBucket << ")\n";
            std::cout << "Selected wallpaper: " << chosen.filePath << "\n";
            std::cout << "Darkness score: " << chosen.score << std::endl;

            executeWallpaperChange(execStr, chosen, hour, chosenBucket);
        }
        catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
