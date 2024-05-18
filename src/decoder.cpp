#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <zlib.h>

int main()
{
    std::string path;
    std::cout << "##########################################################################################" << std::endl;
    std::cout << "#                                                                                        #" << std::endl;
    std::cout << "#                            Parallel Kmeans Images Decoder                              #" << std::endl;
    std::cout << "#                                                                                        #" << std::endl;
    std::cout << "##########################################################################################" << std::endl<< std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "| This is the Decoder to decode the .kc file generated by the Encoder                    |" << std::endl;
    std::cout << "| If you haven't already compressed a file by the Encoder this Decoder is not so useful  |" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl<< std::endl;
    std::cout << "Please enter the global path of the .kc file you want to decode" << std::endl;
    std::cout << "--> ";
    std::getline(std::cin, path); 
    std::cout << std::endl;
    std::string answer = "d";
    while (answer != "y" && answer != "n")
    {
        std::cout << "Do you want to save a copy .jpg of your Compressed Image? [y/n]"<< std::endl;
        std::cout << "--> ";
        std::cin >> answer;
    }

    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }
    
    std::vector<uint8_t> compressedData((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());

    uLong destLen = 100000000; // You may need to adjust this value
    std::vector<uint8_t> buffer(destLen);

    int result = uncompress(buffer.data(), &destLen, compressedData.data(), compressedData.size());
    if (result != Z_OK) {
        std::cerr << "Decompression failed with error code: " << result << std::endl;
        return 2;
    }

    buffer.resize(destLen);

    size_t pos = 0;
    auto readFromBuffer = [&buffer, &pos] (void* data, size_t size)
    {
    uint8_t* byteData = static_cast<uint8_t*>(data);
    std::copy(buffer.begin() + pos, buffer.begin() + pos + size, byteData);
    pos += size;
    };

    int width, height, k;
    readFromBuffer(&width, sizeof(width));
    readFromBuffer(&height, sizeof(height));
    readFromBuffer(&k, sizeof(k));

    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;

    std::vector<std::vector<int>> clustersColors;
    std::vector<cv::Vec3b> pixels;

    for (int i = 0; i < k; ++i) 
    {
        std::vector<int> features(3);
        for (int& feature : features) 
        {
            readFromBuffer(&feature, sizeof(feature));
        }
        clustersColors.push_back(features);
    }

    int bitsPerColor = std::ceil(std::log2(k));

    int bytesPerColor = (bitsPerColor + 7) / 8;

    int n_points = width * height;

    int pointId = 0;

    while (pointId < n_points) 
    {
        uint8_t counter;
        uint8_t clusterId = 0;
        readFromBuffer(&counter, sizeof(counter));
        readFromBuffer(&clusterId, sizeof(clusterId));
        std::cout << "Point " << pointId << " - Cluster " << static_cast<int>(clusterId) << " - Counter " << static_cast<int>(counter) << std::endl;
        for (int i = 0; i < static_cast<int> (counter); ++i) 
        {
            pixels.emplace_back(clustersColors.at(clusterId).at(0), clustersColors.at(clusterId).at(1), clustersColors.at(clusterId).at(2));
            pointId++;
        }
    }

    cv::Mat imageCompressed = cv::Mat(height, width, CV_8UC3);
    for(int y = 0 ; y < height ; y++) 
    {
        for (int x = 0 ; x < width ; x++) 
        {
            imageCompressed.at<cv::Vec3b>(y, x) = pixels.at(y * width + x);
        }
    }
    if(answer == "y")
    {
        if (path.length() >= 3) {
        // Rimuovi gli ultimi 3 caratteri dalla stringa
        path.erase(path.length() - 3);
        }
        std::string outputPath = path + ".jpg";
        cv::imwrite(outputPath, imageCompressed);
    }

    cv::imshow("Compressed Image", imageCompressed);
    cv::waitKey(0);
}