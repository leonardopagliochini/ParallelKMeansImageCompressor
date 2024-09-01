/**
 * @file utilsCLI.hpp
 * @brief Utility functions for the command-line interface
 */

#ifndef UTILSCLI_HPP
#define UTILSCLI_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesUtils.hpp>
#include <boost/process.hpp>

/**
 * @namespace km::utilsCLI
 * @brief Provides utility functions for the command-line interface
 */

namespace km
{
    namespace utilsCLI
    {
        
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