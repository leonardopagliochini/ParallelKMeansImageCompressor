#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <random>
#include <iostream>
#include <omp.h>

#include <thread>
#include "gnuplot-iostream.h"
#include "point.hpp"



class KMeans
{
public:
    KMeans(const int& k, const std::vector<Point> points);

    void run();
    void printClusters() const;
    void plotClusters();
    std::vector<Point> getPoints();
    std::vector<Point> getCentroids();
    int getNumberOfIterationForConvergence();
    int numberOfIterationForConvergence;
private:
    int k;
    std::vector<Point> points;
    std::vector<Point> centroids;
    Gnuplot gp;
};


#endif // KMEANS_HPP