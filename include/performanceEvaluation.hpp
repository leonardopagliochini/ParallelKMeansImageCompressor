#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

class Performance
{
public:
    Performance(); // Default constructor
    void writeCSV(size_t different_colors_size, int k, size_t n_points, double elapsedKmeans, int num_processes = 0);
    static std::string extractFileName(const std::string &outputPath);
    void fillPerformance(int choice, std::string img, std::string method);

private:
    void createOrOpenCSV(const std::string &filename);
    void appendToCSV(std::string filename, int n_diff_colors, int k, int n_colors, const std::string &compType, double time, int num_processes);
    

    std::string img;
    int choice = 0;
    std::string method;
};

Performance::Performance()
{
}

void Performance::fillPerformance(int choice, std::string img, std::string method)
{
    this->choice = choice;
    this->img = img;
    this->method = method;
}

void Performance::writeCSV(size_t different_colors_size, int k, size_t n_points, double elapsedKmeans, int num_processes)
{
    std::string filename = "performanceData.csv";
    createOrOpenCSV(filename);

    std::string compType;
    switch (this->choice)
    {
    case 1:
        compType = "light";
        break;
    case 2:
        compType = "medium";
        break;
    case 3:
        compType = "heavy";
        break;
    default:
        compType = "choice";
        break;
    }

    appendToCSV(filename, different_colors_size, k, n_points, compType, elapsedKmeans, num_processes);
}

void Performance::createOrOpenCSV(const std::string &filename)
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        std::ofstream outfile(filename);
        if (!outfile.is_open())
        {
            std::cerr << "Error creating file!" << '\n';
            return;
        }
        outfile << "img, method, starting colors, k, n_points, comp type, time, world_size" << '\n'; // Write custom header
        outfile.close();                                                                     // Close the file after creating it
    }
}

void Performance::appendToCSV(std::string filename, int startingColors, int remainingColors, int num_points, const std::string &compType, double time, int num_processes)
{
    std::ofstream file(filename, std::ios::app); // Open file for appending
    if (!file.is_open())
    {
        std::cerr << "Error opening file for appending!" << '\n';
        return;
    }
    file << this->img << "," << this->method << "," << startingColors << "," << remainingColors << "," << num_points << "," << compType << "," << time << "," << num_processes << '\n';
}


std::string Performance::extractFileName(const std::string &outputPath)
{
    // Find the position of the last '/' character
    size_t lastSlashPos = outputPath.find_last_of('/');

    // Find the position of the last '.' character
    size_t extensionPos = outputPath.find_last_of('.');

    // If both positions are found and the dot comes after the last slash
    if (lastSlashPos != std::string::npos && extensionPos != std::string::npos && extensionPos > lastSlashPos)
    {
        // Extract the substring between the last '/' and the last '.'
        return outputPath.substr(lastSlashPos + 1, extensionPos - lastSlashPos - 1);
    }

    // If the positions are not found, or the dot is before the last slash, return an empty string
    return "";
}