#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
#include <thread>
#include <memory>
#include <set>
#include <chrono>
#include <cstdint>
#include <zlib.h>

#include <point.hpp>
#include <kMeansMPI.hpp>


#include <mpi.h>


int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int k;
    int n_points;
    std::vector<Point> points;
    std::string path;
    std::string outputPath;
    int height;
    int width;
    std::vector<std::pair<int, Point> > local_points;

     if(rank == 0)
    {
        std::cout << "##########################################################################################" << std::endl;
        std::cout << "#                                                                                        #" << std::endl;
        std::cout << "#                Parallel Kmeans Images Encoder (MPI: did you use mpirun?)               #" << std::endl;
        std::cout << "#                                                                                        #" << std::endl;
        std::cout << "##########################################################################################" << std::endl<< std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "| This program allows you to compress an image by reducing the number of colors in it    |" << std::endl;
        std::cout << "| The output of this Encoder is a file .kc that you need to decode to obtain your image  |" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------" << std::endl<< std::endl;
        std::cout << "Please enter the number of colors you want to keep in the compressed image"<< std::endl;
        std::cout << "--> ";
        std::cin >> k;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Now please enter the global path of the image you want to compress" << std::endl;
        std::cout << "--> ";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::getline(std::cin, path); 
        std::cout << std::endl;
        std::cout << "Choose a name for your output (note that the output will be saved in the outputs directory)" << std::endl;
        std::cout << "You don't need to give any extension" << std::endl;
        std::cout << "--> ";
        std::getline(std::cin, outputPath); 
        outputPath = "outputs/" + outputPath + ".kc";

        cv::Mat image = cv::imread(path);
    // Read an image from file
    

    // Check if the image was loaded successfully
        while (image.empty()) 
        {
            std::cerr << "Error: Unable to load the image." << std::endl;
            std::cout << "Please enter the correct global path of the image you want to compress" << std::endl<< std::endl;
            std::cout << "--> ";
            std::getline(std::cin, path);
            std::cout << std::endl;
            MPI_Bcast(path.data(), path.size() + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            cv::Mat image = cv::imread(path);
        }

        std::cout<< "-------------------------" << path << std::endl;
        height = image.rows;
        width = image.cols;
        //std::set < std::vector<unsigned char> > different_colors; 
        int id = 0;
        for(int y = 0 ; y < height ; y++)
        {
            for (int x = 0 ; x < width ; x++)
            {
                points.emplace_back(Point(id, {image.at<cv::Vec3b>(y, x)[0],image.at<cv::Vec3b>(y, x)[1],image.at<cv::Vec3b>(y, x)[2]}));
                id += 1;
            }
        }

        std::cout << "-----------------------------------------"<< std::endl;
        // std::cout << "Number of different colors in the image: " << different_colors.size() << std::endl;
        // std::cout << "-----------------------------------------"<< std::endl;

        // different_colors.clear();


        n_points = points.size();
    }
    
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int points_per_cluster = n_points / world_size;

    for (int i = 0; i < n_points; i++)
    {
        
        if (rank == 0)
        {
            for (int j = 0; j < world_size; j++)
            {
                int start = j * points_per_cluster;
                int end = (j == world_size - 1) ? n_points : (j + 1) * points_per_cluster;

                if (i >= start && i <  end)
                {
                    MPI_Send(&i, 1, MPI_INT, j, 1, MPI_COMM_WORLD);
                    MPI_Send(&points[i].getFeature(0), 1, MPI_UNSIGNED_CHAR, j, 2, MPI_COMM_WORLD);
                    MPI_Send(&points[i].getFeature(1), 1, MPI_UNSIGNED_CHAR, j, 3, MPI_COMM_WORLD);
                    MPI_Send(&points[i].getFeature(2), 1, MPI_UNSIGNED_CHAR, j, 4, MPI_COMM_WORLD);
                }
            }
        }
        int start = rank * points_per_cluster;
        int end = (rank == world_size - 1) ? n_points : (rank + 1) * points_per_cluster;
        


        if (i >= start && i <  end)
        {
            int id;
            std::vector<int> rgb(3);
            unsigned char r, g, b;
            MPI_Recv(&id, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&r, 1, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&g, 1, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&b, 1, MPI_UNSIGNED_CHAR, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            rgb[0] = static_cast<int>(r);
            rgb[1] = static_cast<int>(g);
            rgb[2] = static_cast<int>(b);
            Point pixel(id, rgb);
            local_points.push_back({0,pixel});
            
        }
        //std::cout << "Rank: " << rank << " Start: " << start << " End: " << end << std::endl;
    
    }

    // if (rank == 0)
    // {
    //     // Get the image dimensions
    //     if(rank == 0)
    //     {local_pointsor<double> rgb = {static_cast<double>(pixels.at(y * width + x)[0]), static_cast<double>(pixels.at(y * width + x)[1]), static_cast<double>(pixels.at(y * width + x)[2])};
    //             int id = y * width + x;
    //             Point pixel(id, rgb);
    //             points.push_back(pixel);

    //             for (int i = 1; i < rank; i++)
    //             {
    //                 if (i*points_per_cluster <= id && id < (i == world_size - 1) ? points.size() : (i + 1) * points_per_cluster)
    //                 {
    //                     MPI_Send(id, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
    //                     MPI_Send(rgb, 3, MPI_DOUBLE, i, id, MPI_COMM_WORLD);
    //                 }
    //             }
    //         }
    //     }

    // }else {
    //     for (int i = 0 ; i < points_per_cluster; i++)
    //     {
    //         int id;
    //         MPI_Recv(&id, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         std::vector<double> rgb(3);
    //         for (int j = 0; j < 3; ++j)
    //         {
    //             MPI_Recv(&rgb[j], 1, MPI_DOUBLE, 0, 4+j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         }
    //         Point pixel(id, rgb);
    //         points.push_back(pixel);
    //     }
    // }

    

    auto start = std::chrono::high_resolution_clock::now();
    if(rank == 0)
    {
        std::cout << "Starting the Compression..." << std::endl;
    }

    std::unique_ptr<KMeans> kmeans;



    if (rank == 0)
    {
        kmeans = std::unique_ptr<KMeans>(new KMeans(k,rank,3, points));
    }else{
        kmeans = std::unique_ptr<KMeans>(new KMeans(k,rank,3));
    }
    kmeans->run(rank, world_size,local_points);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Display the image

    if(rank == 0)
    {   
        std::cout << "Compression done in " << elapsed.count() << std::endl;
        std::cout << std::endl;
        std::cout << "Saving the Compressed Image..." << std::endl;
        std::ofstream outputFile(outputPath, std::ios::app);
        std::vector<uint8_t> buffer;

        // Helper function to write data to buffer
    auto writeToBuffer = [&buffer](const void* data, size_t size) {
        const uint8_t* byteData = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), byteData, byteData + size);
    };

    // Scrivi width, height e k
    writeToBuffer(&width, sizeof(width));
    writeToBuffer(&height, sizeof(height));
    writeToBuffer(&k, sizeof(k));

    // Scrivi i centroidi
    for (Point& centroid : kmeans->getCentroids()) 
    {
        for (int i = 0; i < 3; ++i) 
        {
            int feature = centroid.getFeature_int(i);
            writeToBuffer(&feature, sizeof(feature));
        }
    }

    // Determina il numero di bit necessari per rappresentare k colori
    int bitsPerColor = std::ceil(std::log2(k));
    int bytesPerColor = (bitsPerColor + 7) / 8; // Arrotonda per eccesso

    // Scrivi i clusterId per ogni punto usando il numero minimo di byte
    for (Point &p : kmeans->getPoints()) 
    {
        uint32_t clusterId = p.clusterId;
        for (int byte = 0; byte < bytesPerColor; ++byte) 
        {
            uint8_t currentByte = (clusterId >> (byte * 8)) & 0xFF;
            writeToBuffer(&currentByte, sizeof(currentByte));
        }
    }

    // Comprimi il buffer usando zlib
    uLong sourceLen = buffer.size();
    uLong destLen = compressBound(sourceLen);
    std::cout << "SourceLen: " << sourceLen << std::endl;
    std::vector<uint8_t> compressedData(destLen);

    int result = compress(compressedData.data(), &destLen, buffer.data(), sourceLen);
    if (result != Z_OK) 
    {
        std::cerr << "Compression failed with error code: " << result << std::endl;
    }

    // Ridimensiona il vettore al reale numero di byte compressi
    compressedData.resize(destLen);

    // Scrivi i dati compressi su file
    outputFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

        // outputFile  << width << ","<< height << "," << k << std::endl;
        // for (int i = 0 ; i < k ; i++)
        // {
        //     Point centroid = kmeans->getCentroids()[i];
        //     outputFile << centroid.features[0];
        //     outputFile << ",";
        //     outputFile << centroid.features[1];
        //     outputFile << ",";
        //     outputFile << centroid.features[2] << std::endl;
        // }

        // for (Point &p : kmeans->getPoints())
        // {
        //     outputFile << p.clusterId << std::endl;
        // }    

        // cv::Mat imageCompressed = cv::Mat(height, width, CV_8UC3);
        // for(int y = 0 ; y < height ; y++)
      
        // cv::Mat imageCompressed = cv::Mat(height, width, CV_8UC3);
        // for(int y = 0 ; y < height ; y++)
        // {
        //     for (int x = 0 ; x < width ; x++)
        //     {
        //         imageCompressed.at<cv::Vec3b>(y, x) = pixels.at(y * width + x);
        //     }
        // }
        // std::string outputPath = "outputImages/imageCompressed_" + std::to_string(k) + "_colors.jpg";
        // cv::imwrite(outputPath, imageCompressed);

        std::ofstream perfEval("performanceEvaluation.txt", std::ios::app);
        perfEval << argv[0] << std::endl;
        perfEval << "--------------------------------------"<< std::endl;
        perfEval << "Image path: " << path << std::endl;
        perfEval << "Number of colors: " << k << std::endl;
        perfEval << "Number of points: " << points.size() << std::endl;
        perfEval << "Number of iterations: " << kmeans->numberOfIterationForConvergence << std::endl;
        perfEval << "Number of Processors: " << world_size << std::endl;
        perfEval << " ==>" << "Time: " << elapsed.count() << "s" << std::endl;
        perfEval << std::endl;
        perfEval << std::endl;
        perfEval << std::endl;

        // cv::imshow("Original", image);
        // cv::waitKey(1);
        // cv::imshow("Compressed", imageCompressed);
        // cv::waitKey(0);

        std::cout << std::endl;
        std::cout << "Work done!" << std::endl;
        
    }
    MPI_Finalize();
}
