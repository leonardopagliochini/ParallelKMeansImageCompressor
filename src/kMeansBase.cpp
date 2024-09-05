/**
 * @file kMeansBase.cpp
 * @brief Base class implementation for the K-means clustering algorithm.
 * @details This file contains the implementation of the `KMeansBase` class, which serves as the base class
 *          for different versions of the K-means clustering algorithm. It provides fundamental functionalities
 *          such as initialization of centroids, getters for points, centroids, and the number of iterations.
 */


#include "kMeansBase.hpp"
#include <random>
#include <chrono>
#include <iostream>

km::KMeansBase::KMeansBase(const int& k, const std::vector<km::Point>& points)
    : k(k), points(points), number_of_iterations(0)
{
    size_t size = points.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, size - 1);

    for (int i = 0; i < k; ++i)
    {
        centroids.emplace_back(points[dis(gen)]);
    }
}

km::KMeansBase::KMeansBase(const int& k)
    : k(k), number_of_iterations(0)
{
    this->centroids = std::vector<Point>(k, Point());
}

std::vector<km::Point> km::KMeansBase::getPoints() const
{
    return points;
}

std::vector<km::Point> km::KMeansBase::getCentroids() const
{
    return centroids;
}

int km::KMeansBase::getIterations() const
{
    return number_of_iterations;
}
