#pragma once
#include <string>
#include <vector>

extern std::vector<std::string> supportedExtensions;
bool isSupportedFormat(const std::string& filename);
size_t scanFolder(std::vector<std::string>& imageFiles, const std::string& folderPath);
std::string formatTime(int seconds);
size_t getImages(std::vector<std::string>& images, const std::string& inputPath);

namespace Cursor {
    void termClear();
    void reset();
    void hide();
    void show();
    void cr();
}; // namespace Cursor

std::vector<std::string> csv_split(const std::string& line, char delimiter);
std::string trim(const std::string& str);
void setNonBlockingInput(bool enable);
bool checkKeyPress(char* c);
bool executeCommand(const std::string& program, const std::string& filePath);
bool fs_exists(const std::string& path);
