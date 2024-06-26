/**
 * @file kMeansCUDA.hpp
 * @brief Implementation of the K-means clustering algorithm using CUDA
 */

#ifndef KMEANS_CUDA_HPP
#define KMEANS_CUDA_HPP

#include <random>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "point.hpp"

/**
 * @class KMeans
 * @brief Represents the K-means clustering algorithm using CUDA
 */

namespace km
{
    class KMeans
    {
    public:
        /**
         * @brief Constructor for KMeans
         * @param k Number of clusters
         * @param points Vector of points
         */
        KMeans(const int& k, const std::vector<Point> &points);

        /**
         * @brief Runs the K-means clustering algorithm using CUDA
         */
        void run();

        /**
         * @brief Prints the clusters
         */
        void printClusters() const;

        /**
         * @brief Plots the clusters
         */
        void plotClusters();

        /**
         * @brief Gets the points
         * @return Vector of points
         */
        std::vector<Point> getPoints();

        /**
         * @brief Gets the centroids
         * @return Vector of centroids
         */
        std::vector<Point> getCentroids();

        /**
         * @brief Gets the number of iterations
         * @return Number of iterations
         */
        auto getIterations() -> int;

    private:
        int k; ///< Number of clusters
        std::vector<Point> points; ///< Vector of points
        std::vector<Point> centroids; ///< Vector of centroids
        int number_of_iterations; ///< Number of iterations
    };
} // namespace km

#endif // KMEANS_CUDA_HPP