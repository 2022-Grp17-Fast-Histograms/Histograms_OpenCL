#pragma once

#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>

void CalculateAverage(std::vector<int> imageVector, int globalOffset, int imageWidth, int numOfBlocks, int blockSize, int blockWidth, int blockHeight, std::vector<double> &average) {
    int x = 0; //X COORD
    int y = 0; //Y COORD
    for (int block = 0; block < numOfBlocks; block++) {
        int i = 0; //ROW (Y)
        int j = 0; //COL (X)
        int offset = x + y * imageWidth + globalOffset;
        int blocksum = 0;
        for (int pixel = 0; pixel < blockSize; pixel++) {
            blocksum += imageVector[i * imageWidth + j + offset];
            j++;
            if (j == blockWidth) {
                j = 0;
                i++;
            }
        }
        average[block] = (double)blocksum / blockSize;
        x += blockWidth;
        if (x >= imageWidth) {
            x = 0;
            y += blockHeight;
        }
    }
}

void CalculateVariance(std::vector<int> imageVector, int globalOffset, int imageWidth, int numOfBlocks, int blockSize, int blockWidth, int blockHeight, std::vector<double> average, std::vector<double> &variance) {
    int x = 0; //X COORD
    int y = 0; //Y COORD
    for (int block = 0; block < numOfBlocks; block++) {
        int i = 0; //ROW (Y)
        int j = 0; //COL (X)
        int offset = x + y * imageWidth + globalOffset;
        double variancesum = 0;
        for (int pixel = 0; pixel < blockSize; pixel++) {
            int val = imageVector[i * imageWidth + j + offset];
            double var = std::pow((val - average[block]), 2);
            variancesum += var;
            j++;
            if (j == blockWidth) {
                j = 0;
                i++;
            }
        }
        variance[block] = (double)variancesum / blockSize;
        x += blockWidth;
        if (x >= imageWidth) {
            x = 0;
            y += blockHeight;
        }
    }
}

void CalculateAverageAndVariance(std::vector<int> imageVector, int globalOffset, int imageWidth, int numOfBlocks, int blockSize, int blockWidth, int blockHeight, std::vector<double> &average, std::vector<double> &variance) {
    int x = 0; //X COORD
    int y = 0; //Y COORD
    for (int block = 0; block < numOfBlocks; block++) {
        int i = 0; //ROW (Y)
        int j = 0; //COL (X)
        int offset = x + y * imageWidth + globalOffset;
        int blocksum = 0;
        for (int pixel = 0; pixel < blockSize; pixel++) {
            blocksum += imageVector[i * imageWidth + j + offset];
            j++;
            if (j == blockWidth) {
                j = 0;
                i++;
            }
        }
        average[block] = (double)blocksum / blockSize;
        double variancesum = 0;
        i = 0;
        j = 0;
        for (int pixel = 0; pixel < blockSize; pixel++) {
            int val = imageVector[i * imageWidth + j + offset];
            double var = std::pow((val - average[block]), 2);
            variancesum += var;
            j++;
            if (j == blockWidth) {
                j = 0;
                i++;
            }
        }
        variance[block] = (double)variancesum / blockSize;
        x += blockWidth;
        if (x >= imageWidth) {
            x = 0;
            y += blockHeight;
        }
    }
}

void CalculateHistogram(std::vector<double> input, int numOfBins, std::vector<int> &bins) {
    int binSize = 256/numOfBins;

    for (int i = 0; i < input.size(); i++) {
        int interval = input[i]/binSize;
        bins[interval]++;
    }
}

void CalculateHistogram(std::vector<double> input, int numOfBins, std::vector<int> &bins, std::vector<double> increment) {
    int binSize = 256/numOfBins;

    for (int i = 0; i < input.size(); i++) {
        int interval = input[i]/binSize;
        bins[interval] += increment[i];
    }
}

template <typename T>
bool ValidateVector(std::vector<T> input, std::vector<T> validatingVector) {
    bool result = false;
    for (int i = 0; i < input.size(); i++) {
        if (input[i] != validatingVector[i]){
            break;
        }
        if (i+1 == input.size()){
            result = true;
        }
    }
    return result;
}
