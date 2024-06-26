/**
 * @file utilsCLI.hpp
 * @brief Utility functions for the command-line interface
 */

#ifndef UTILSCLI_HPP
#define UTILSCLI_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesUtils.hpp>

/**
 * @class UtilsCLI
 * @brief Provides utility functions for the command-line interface
 */

namespace km
{
    namespace utilsCLI
    {

        /**
         * @brief Displays the header for the sequential encoder
         */
         void sequentialEncoderHeader();

        /**
         * @brief Displays the header for the MPI encoder
         */
         void mpiEncoderHeader();

        /**
         * @brief Displays the header for the OpenMP encoder
         */
         void ompEncoderHeader();
        /**
         * @brief Displays the main menu header
         */
         void mainMenuHeader();

        /**
         * @brief Displays the decoder header
         */
         void decoderHeader();

        /**
         * @brief Displays the work done message
         */
         void workDone();

        /**
         * @brief Handles the compression choices
         * @param levelsColorsChoice Choice of color levels
         * @param typeCompressionChoice Choice of compression type
         * @param outputPath Output path
         * @param image Input image
         * @param executionStandard Execution standard
         */
         void compressionChoices(int &levelsColorsChoice, int &typeCompressionChoice, std::string &outputPath, cv::Mat &image, int executionStandard);

        /**
         * @brief Prints the compression information
         * @param originalWidth Original width of the image
         * @param originalHeight Original height of the image
         * @param width Width of the compressed image
         * @param height Height of the compressed image
         * @param k Number of clusters
         * @param different_colors_size Number of different colors
         */
         void printCompressionInformations(int &originalWidth, int &originalHeight, int &width, int &height, int &k, size_t &different_colors_size);

        /**
         * @brief Displays the decoding menu
         * @param path Path of the directory containing the compressed images
         * @param imageNames Vector of image names
         * @param decodeDir Path of the decoding directory
         */
         void displayDecodingMenu(std::string &path, std::vector<std::filesystem::path> &imageNames, std::filesystem::path &decodeDir);
    };
}

#endif // UTILSCLI_HPP