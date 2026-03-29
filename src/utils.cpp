
#include "utils.hpp"

#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <termios.h>
#include <vector>

std::vector<std::string> supportedExtensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"};

bool isSupportedFormat(const std::string& filename)
{
    std::string extension = filename.substr(filename.find_last_of("."));
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(supportedExtensions.begin(), supportedExtensions.end(), extension) != supportedExtensions.end();
}

size_t scanFolder(std::vector<std::string>& imageFiles, const std::string& folderPath)
{
    std::cout << "Scanning folder: " << folderPath << std::endl;

    if (!fs_exists(folderPath)) {
        std::cerr << "Error scanning folder: " << folderPath << std::endl;
        return 0;
    }

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
            if (entry.is_regular_file() && isSupportedFormat(entry.path().filename().string())) {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Error scanning folder: " << ex.what() << std::endl;
        exit(1);
        return 0;
    }

    size_t totalCount = imageFiles.size();
    std::cout << "Found " << totalCount << " image files." << std::endl;

    if (totalCount == 0) {
        std::cout << "No images found." << std::endl;
    }

    return totalCount;
}

size_t getImages(std::vector<std::string>& images, const std::string& inputPath)
{

    if (std::filesystem::is_regular_file(inputPath)) {
        images.push_back(inputPath);
    }
    else if (std::filesystem::is_directory(inputPath)) {
        scanFolder(images, inputPath);
    }
    return images.size();
}

std::string formatTime(int seconds)
{
    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int secs = seconds % 60;

    if (hours > 0) {
        return std::to_string(hours) + "h " + std::to_string(minutes) + "m " + std::to_string(secs) + "s";
    }
    else if (minutes > 0) {
        return std::to_string(minutes) + "m " + std::to_string(secs) + "s";
    }
    else {
        return std::to_string(secs) + "s";
    }
}

// clang-format off
namespace Cursor {
    void termClear() { std::cout << "\033[2J"; }
    void reset() { std::cout << "\033[H"; }
    void hide() { std::cout << "\033[?25l" << std::flush; }
    void show() { std::cout << "\033[?25h" << std::flush; }
    void cr() { std::cout << "\r" << std::flush; }

}; // namespace Cursor
// clang-format on

std::vector<std::string> csv_split(const std::string& line, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(line);
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void setNonBlockingInput(bool enable)
{
    static struct termios old_tio, new_tio;
    static bool initialized = false;

    if (enable) {
        if (!initialized) {
            tcgetattr(STDIN_FILENO, &old_tio);
            new_tio = old_tio;
            new_tio.c_lflag &= ~(ICANON | ECHO);
            initialized = true;
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);

        // Set stdin to non-blocking
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }
    else {
        if (initialized) {
            tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);

            // Restore blocking mode
            int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
            fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
        }
    }
}

bool checkKeyPress(char* c)
{
    int n = read(STDIN_FILENO, c, 1);
    return n > 0;
}

std::string trim(const std::string& str)
{
    const std::string whitespace = " \n\r\t\f\v";
    const auto first = str.find_first_not_of(whitespace);
    if (first == std::string::npos) return "";
    const auto last = str.find_last_not_of(whitespace);
    return str.substr(first, (last - first + 1));
}

bool executeCommand(const std::string& program, const std::string& filePath)
{
    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "Fork failed" << std::endl;
        return false;
    }

    if (pid == 0) {
        // Child process
        // Redirect stdout/stderr to /dev/null in child to avoid issues
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull != -1) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }

        // Execute the command directly
        execlp(program.c_str(), program.c_str(), filePath.c_str(), nullptr);

        // If execlp returns, it failed
        exit(EXIT_FAILURE);
    }
    else {
        // Parent process
        int status;
        waitpid(pid, &status, 0);

        int exitCode = WEXITSTATUS(status);
        if (exitCode != 0) {
            std::cerr << "Command exited with status: " << exitCode << std::endl;
            return false;
        }
        return true;
    }
}

bool fs_exists(const std::string& path)
{
    try {
        return std::filesystem::exists(path);
    }
    catch (...) {
    }
    return false;
}
