/**
 * @file performanceEvaluation.cpp
 * @brief Utility functions for evaluating and recording the performance of image compression algorithms.
 * @details This file contains the implementation of the `Performance` class, which provides functionalities 
 *          to record performance metrics such as time taken for compression, number of iterations, and other 
 *          relevant statistics. The class includes methods to fill performance data, write results to a CSV file, 
 *          and manage the creation and appending of data to the CSV file for easy analysis.
 */



#include <performanceEvaluation.hpp>
#include <utility>

km::Performance::Performance() = default;

auto km::Performance::fillPerformance(const int& choice, const std::string &img, const std::string &method) -> void
{
    this->choice = choice;
    this->img = std::move(img);
    this->method = std::move(method);
}

auto km::Performance::writeCSV(const int& different_colors_size, const int& k, const int& n_points, const double& elapsedKmeans, const int& number_of_iterations, const int& num_processes) -> void
{
    std::string filename = "./performanceData.csv";
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
    appendToCSV(filename, different_colors_size, k, n_points, compType, elapsedKmeans, number_of_iterations, num_processes);
}

auto km::Performance::createOrOpenCSV(const std::string &filename) -> void
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        std::ofstream outfile(filename);
        if (!outfile.is_open())
        {
            std::cerr << "Error creating file!" << std::endl;
            return;
        }
        outfile << "img,method,starting colors,k,n_points,comp type,time,world_size" << std::endl; // Write custom header
        outfile.close(); // Close the file after creating it
    }
}

auto km::Performance::appendToCSV(const std::string &filename, const int& startingColors, const int& remainingColors, const int& num_points, const std::string &compType, const double& time, const int& number_of_iterations, const int& num_processes) -> void
{
    std::ofstream file(filename, std::ios::app); // Open file for appending
    if (!file.is_open())
    {
        std::cerr << "Error opening file for appending!" << std::endl;
        return;
    }
    double time_per_iteration = time / number_of_iterations;
    file << this->img << "," << this->method << "," << startingColors << "," << remainingColors << "," << num_points << "," << compType << "," << time_per_iteration << "," << num_processes << std::endl;
}

auto km::Performance::extractFileName(const std::string &outputPath) -> std::string
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