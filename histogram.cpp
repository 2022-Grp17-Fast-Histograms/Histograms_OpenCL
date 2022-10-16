#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <utility>
#include <CL/opencl.hpp>
#include "utilities.hpp"
#include "histogram_cpu.hpp"

// Show all debug messages
const bool DEBUG_MODE_CPU = true;
const bool DEBUG_MODE_GPU = true;

// Choose tests to run
const bool ALL_CHANNELS = true;                             // TRUE = CALCULATE FOR ALL CHANNELS; FALSE = CALCULATE ONLY FOR LUMA (Y)
const bool AVERAGE_TEST = true;                             // AVERAGE ONLY
const bool VARIANCE_TEST_1 = true;                          // AVERAGE + VARIANCE (ON SAME KERNEL)
const bool VARIANCE_TEST_2 = true;                          // AVERAGE THEN VARIANCE (2 KERNELS)
const bool AVERAGE_HISTOGRAM_TEST_1 = true;                 // AVERAGE + AVERAGE HISTOGRAM (ON SAME KERNEL)
const bool AVERAGE_HISTOGRAM_TEST_2 = true;                 // AVERAGE THEN AVERAGE HISTOGRAM (2 KERNELS)
const bool VARIANCE_HISTOGRAM_TEST_1 = true;               // AVERAGE + VARIANCE + VARIANCE HISTOGRAMS (ON SAME KERNEL)

// Choose outputs to be generated
const bool GENERATE_AVERAGE_IMG = true;                     // NEEDS AVERAGE TEST 1
const bool GENERATE_VARIANCE_IMG = true;                    // NEEDS VARIANCE TEST 1
const bool EXPORT_AVERAGE_HISTOGRAM = true;                 // NEEDS AVERAGE HISTOGRAM TEST 1 FOR GPU HISTOGRAM
const bool EXPORT_VARIANCE_HISTOGRAM = true;                // NEEDS VARIANCE HISTOGRAM TEST 1 FOR GPU HISTOGRAM

// Set path for input image
const std::string FILEPATH = "Input Images/DOTA2_I420_1920x1080.yuv"; 

// Set width height
const int IMG_WIDTH = 1920;
const int IMG_HEIGHT = 1080;

// Set block size
const int BLOCK_WIDTH = 8;
const int BLOCK_HEIGHT = 8;

// Set number of bins for the histograms
const int NUM_OF_BINS = 16;

int main(int argc, char const *argv[])
{
    std::cout << "Using image file: " << FILEPATH << std::endl << std::endl;

    // Calculate channel sizes for each frame
    int y_Size = IMG_WIDTH * IMG_HEIGHT;
    int u_Size = (IMG_WIDTH/2) * (IMG_HEIGHT/2);
    int v_Size = (IMG_WIDTH/2) * (IMG_HEIGHT/2);
    int imageSize = y_Size + u_Size + v_Size;
    if (DEBUG_MODE_CPU) {
        std::cout << "Y SIZE: " << y_Size << std::endl;
        std::cout << "U SIZE: " << u_Size << std::endl;
        std::cout << "V SIZE: " << v_Size << std::endl;
        std::cout << "Image file size: " << imageSize << std::endl;
    }
    
    // Calculate blocksize and numbers of blocks for each channel;
    int y_BlockWidth = BLOCK_WIDTH;
    int y_BlockHeight = BLOCK_HEIGHT;
    int y_BlockSize = y_BlockWidth * y_BlockHeight;
    int y_NumOfBlocks = y_Size / y_BlockSize;
    if (DEBUG_MODE_CPU) {
	    std::cout << "Y BLOCK SIZE: " << y_BlockSize << std::endl;
        std::cout << "Y NUM OF BLOCKS: " << y_NumOfBlocks << std::endl;
    }

    int u_BlockWidth = BLOCK_WIDTH/2;
    int u_BlockHeight = BLOCK_HEIGHT/2;
    int u_BlockSize = u_BlockWidth * u_BlockHeight;
    int u_NumOfBlocks = u_Size / u_BlockSize;
    if (DEBUG_MODE_CPU) {
	    std::cout << "U BLOCK SIZE: " << u_BlockSize << std::endl;
        std::cout << "U NUM OF BLOCKS: " << u_NumOfBlocks << std::endl;
    }

    int v_BlockWidth = BLOCK_WIDTH/2;
    int v_BlockHeight = BLOCK_HEIGHT/2;
    int v_BlockSize = v_BlockWidth * v_BlockHeight;
    int v_NumOfBlocks = v_Size / v_BlockSize;
    if (DEBUG_MODE_CPU) {
	    std::cout << "V BLOCK SIZE: " << v_BlockSize << std::endl;
        std::cout << "V NUM OF BLOCKS: " << v_NumOfBlocks << std::endl;
    }

    // Open file
    std::ifstream inputYUV (FILEPATH, std::ios::binary);
    if (!inputYUV.is_open()) {        
        std::cout << "Error opening file " << FILEPATH << std::endl;
        return(0);        
    }
    inputYUV.seekg(0, inputYUV.end);
    int fSize = inputYUV.tellg();
    if (DEBUG_MODE_CPU) {
        std::cout << "Read File Size: " << fSize << std::endl;
    }
    inputYUV.clear();
    inputYUV.seekg(inputYUV.beg);

    int actualSize = std::filesystem::file_size(FILEPATH);
    if (DEBUG_MODE_CPU) {
        std::cout << "Actual file size: " << actualSize << std::endl;
    }

    if (fSize != actualSize) {
        std::cout << "Size read different than actual file size"<< std::endl;
        inputYUV.close();
        return(0);   
    }

    if (fSize != imageSize) {
        std::cout << "Size read different than image file size" << std::endl;
        inputYUV.close();
        return(0);   
    }

    // Read file into vector
    std::vector<int> imageVector(imageSize);
    for (int i = 0; i < imageSize; i++) {
        imageVector.at(i) = inputYUV.get();
    }
    inputYUV.close();

    if (DEBUG_MODE_CPU) {
        std::cout << "\n================IMAGE AND BLOCK CONFIGURATION=================\n\n";

        std::cout << "Image dimensions: " << IMG_WIDTH << "x" << IMG_HEIGHT << std::endl;
        std::cout << "Block dimensions: " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << std::endl;
        std::cout << "Number of bins:" << NUM_OF_BINS << std::endl;
    }

    std::cout << "\n=============================CPU==============================\n\n";

    // Create Output vectors for Average, Average Bins (histogram),
    // Variance, Variance Bins(histogram) for each channel
    std::vector<double> y_AverageCPU(y_NumOfBlocks);
    std::vector<int> y_AverageBinsCPU(NUM_OF_BINS);
    std::vector<double> y_VarianceCPU(y_NumOfBlocks);
    std::vector<int> y_VarianceBinsCPU(NUM_OF_BINS);

    std::vector<double> u_AverageCPU(u_NumOfBlocks);
    std::vector<int> u_AverageBinsCPU(NUM_OF_BINS);
    std::vector<double> u_VarianceCPU(u_NumOfBlocks);
    std::vector<int> u_VarianceBinsCPU(NUM_OF_BINS);

    std::vector<double> v_AverageCPU(v_NumOfBlocks);
    std::vector<int> v_AverageBinsCPU(NUM_OF_BINS);
    std::vector<double> v_VarianceCPU(v_NumOfBlocks);
    std::vector<int> v_VarianceBinsCPU(NUM_OF_BINS);

    // Create Timer Variables
    double y_ElapsedTimeAverageCPU, u_ElapsedTimeAverageCPU, v_ElapsedTimeAverageCPU;
    double y_ElapsedTimeVarianceCPU, u_ElapsedTimeVarianceCPU, v_ElapsedTimeVarianceCPU;
    double y_ElapsedTimeAverageHistCPU, u_ElapsedTimeAverageHistCPU, v_ElapsedTimeAverageHistCPU;
    double y_ElapsedTimeVarianceHistCPU, u_ElapsedTimeVarianceHistCPU, v_ElapsedTimeVarianceHistCPU;

    std::cout << "\n--------------------------AVERAGES----------------------------\n\n";
    // Average of Channel Y
    TimeInterval timer("milli");
    CalculateAverage(imageVector, 0, IMG_WIDTH, y_NumOfBlocks, y_BlockSize, y_BlockWidth, y_BlockHeight, y_AverageCPU);
    y_ElapsedTimeAverageCPU = timer.Elapsed();
    std::cout << "Elapsed time Y Channel Average (ms) = " << y_ElapsedTimeAverageCPU << std::endl;

    if (ALL_CHANNELS) {
        // Average of Channel U
        timer = TimeInterval("milli");
        CalculateAverage(imageVector, y_Size, IMG_WIDTH/2, u_NumOfBlocks, u_BlockSize, u_BlockWidth, u_BlockHeight, u_AverageCPU);
        u_ElapsedTimeAverageCPU = timer.Elapsed();
        std::cout << "Elapsed time U Channel Average (ms) = " << u_ElapsedTimeAverageCPU << std::endl;

        // Average of Channel V
        timer = TimeInterval("milli");
        CalculateAverage(imageVector, y_Size + u_Size, IMG_WIDTH/2, v_NumOfBlocks, v_BlockSize, v_BlockWidth, v_BlockHeight, v_AverageCPU);
        v_ElapsedTimeAverageCPU = timer.Elapsed();
        std::cout << "Elapsed time V Channel Average (ms) = " << v_ElapsedTimeAverageCPU << std::endl;
    }

    std::cout << "\n--------------------------VARIANCES---------------------------\n\n";
    // Variance of Channel Y
    timer = TimeInterval("milli");
    CalculateVariance(imageVector, 0, IMG_WIDTH, y_NumOfBlocks, y_BlockSize, y_BlockWidth, y_BlockHeight, y_AverageCPU, y_VarianceCPU);
    y_ElapsedTimeVarianceCPU = timer.Elapsed();
    std::cout << "Elapsed time Y Channel Variance (ms) = " << y_ElapsedTimeVarianceCPU << std::endl;

    if (ALL_CHANNELS) {
        // Average of Channel U
        timer = TimeInterval("milli");
        CalculateVariance(imageVector, y_Size, IMG_WIDTH/2, u_NumOfBlocks, u_BlockSize, u_BlockWidth, u_BlockHeight, u_AverageCPU, u_VarianceCPU);
        u_ElapsedTimeVarianceCPU = timer.Elapsed();
        std::cout << "Elapsed time U Channel Variance (ms) = " << u_ElapsedTimeVarianceCPU << std::endl;

        // Average of Channel V
        timer = TimeInterval("milli");
        CalculateVariance(imageVector, y_Size + u_Size, IMG_WIDTH/2, v_NumOfBlocks, v_BlockSize, v_BlockWidth, v_BlockHeight, v_AverageCPU, v_VarianceCPU);
        v_ElapsedTimeVarianceCPU = timer.Elapsed();
        std::cout << "Elapsed time V Channel Variance (ms) = " << v_ElapsedTimeVarianceCPU << std::endl;
    }

    std::cout << "\n-------------------------HISTOGRAMS---------------------------\n\n";
    // Average Histogram of Channel Y
    timer = TimeInterval("milli");
    CalculateHistogram(y_AverageCPU, NUM_OF_BINS, y_AverageBinsCPU);
    y_ElapsedTimeAverageHistCPU = timer.Elapsed();
    std::cout << "Elapsed time Y Channel Average Hist (ms) = " << y_ElapsedTimeAverageHistCPU << std::endl;

    if (ALL_CHANNELS) {
        // Average Histogram of Channel U
        timer = TimeInterval("milli");
        CalculateHistogram(u_AverageCPU, NUM_OF_BINS, u_AverageBinsCPU);
        u_ElapsedTimeAverageHistCPU = timer.Elapsed();
        std::cout << "Elapsed time U Channel Average Hist (ms) = " << u_ElapsedTimeAverageHistCPU << std::endl;

        // Average Histogram of Channel V
        timer = TimeInterval("milli");
        CalculateHistogram(v_AverageCPU, NUM_OF_BINS, v_AverageBinsCPU);
        v_ElapsedTimeAverageHistCPU = timer.Elapsed();
        std::cout << "Elapsed time V Channel Average Hist (ms) = " << v_ElapsedTimeAverageHistCPU << std::endl;
    }

    // Variance Histogram of Channel Y
    timer = TimeInterval("milli");
    CalculateHistogram(y_AverageCPU, NUM_OF_BINS, y_VarianceBinsCPU, y_VarianceCPU);
    y_ElapsedTimeVarianceHistCPU = timer.Elapsed();
    std::cout << "Elapsed time Y Channel Variance Hist (ms) = " << y_ElapsedTimeVarianceHistCPU << std::endl;
    
    if (ALL_CHANNELS) {
        // Variance Histogram of Channel U
        timer = TimeInterval("milli");
        CalculateHistogram(u_AverageCPU, NUM_OF_BINS, u_VarianceBinsCPU, u_VarianceCPU);
        u_ElapsedTimeVarianceHistCPU = timer.Elapsed();
        std::cout << "Elapsed time U Channel Variance Hist (ms) = " << u_ElapsedTimeVarianceHistCPU << std::endl;

        // Variance Histogram of Channel V
        timer = TimeInterval("milli");
        CalculateHistogram(v_AverageCPU, NUM_OF_BINS, v_VarianceBinsCPU, v_VarianceCPU);
        v_ElapsedTimeVarianceHistCPU = timer.Elapsed();
        std::cout << "Elapsed time V Channel Variance Hist (ms) = " << v_ElapsedTimeVarianceHistCPU << std::endl;
    }

    std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
    if (ALL_CHANNELS) {
        std::cout << "Elapsed time Average (Y + U + V) (ms) = " << y_ElapsedTimeAverageCPU + u_ElapsedTimeAverageCPU + v_ElapsedTimeAverageCPU << std::endl;
        std::cout << "Elapsed time Variance (Y + U + V) (ms) = " << y_ElapsedTimeVarianceCPU + u_ElapsedTimeVarianceCPU + v_ElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Average Hist (Y + U + V) (ms) = " << y_ElapsedTimeAverageHistCPU + u_ElapsedTimeAverageHistCPU + v_ElapsedTimeAverageHistCPU << std::endl;
        std::cout << "Elapsed time Variance Hist (Y + U + V) (ms) = " << y_ElapsedTimeVarianceHistCPU + u_ElapsedTimeVarianceHistCPU + v_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel Y (Avg + Var) (ms) = " << y_ElapsedTimeAverageCPU + y_ElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel U (Avg + Var) (ms) = " << u_ElapsedTimeAverageCPU + u_ElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel V (Avg + Var) (ms) = " << v_ElapsedTimeAverageCPU + v_ElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time Channel Y (Avg + Var + Hist) (ms) = " << y_ElapsedTimeAverageCPU + y_ElapsedTimeVarianceCPU + y_ElapsedTimeAverageHistCPU + y_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel U (Avg + Var + Hist) (ms) = " << u_ElapsedTimeAverageCPU + u_ElapsedTimeVarianceCPU + u_ElapsedTimeAverageHistCPU + u_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel V (Avg + Var + Hist) (ms) = " << v_ElapsedTimeAverageCPU + v_ElapsedTimeVarianceCPU + v_ElapsedTimeAverageHistCPU + v_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Total Elapsed time (ms) = " << y_ElapsedTimeAverageCPU + y_ElapsedTimeVarianceCPU + y_ElapsedTimeAverageHistCPU + y_ElapsedTimeVarianceHistCPU + u_ElapsedTimeAverageCPU + u_ElapsedTimeVarianceCPU + u_ElapsedTimeAverageHistCPU + u_ElapsedTimeVarianceHistCPU + v_ElapsedTimeAverageCPU + v_ElapsedTimeVarianceCPU + v_ElapsedTimeAverageHistCPU + v_ElapsedTimeVarianceHistCPU << std::endl;
    }
    else {
        std::cout << "Elapsed time (Avg + Var) (ms) = " << y_ElapsedTimeAverageCPU + y_ElapsedTimeVarianceCPU << std::endl;
        std::cout << "Elapsed time (Avg + Var + Hist) (ms) = " << y_ElapsedTimeAverageCPU + y_ElapsedTimeVarianceCPU << std::endl;
    }

    std::cout << "\n=============================GPU==============================\n\n";

    // Get platform and device information
    int err;
    cl::Platform platform = cl::Platform::getDefault();
    std::cout << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    std::cout << "Device name: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create context
    cl::Context context(devices, NULL, NULL, NULL, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Context ERROR: " << err << std::endl;
    }

    // Create CommandQueue
    cl::CommandQueue commandQueue(context, devices[0], cl::QueueProperties::Profiling, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Queue ERROR: " << err << std::endl;
    }

    // Read Program Source
    std::ifstream sourceFile("histogram_kernel.cl");
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));

    // Create Program
    cl::Program program(context, sourceCode, err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Program ERROR: " << err << std::endl;
    }

    // Build Program
    err = program.build(devices);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Build Program ERROR: " << err << std::endl;
    }

    // Create Kernels
    cl::Kernel averageKernel(program, "CalculateAverage");
    cl::Kernel varianceKernel(program, "CalculateVariance");
    cl::Kernel averageVarianceKernel(program, "CalculateAverageAndVariance");
    cl::Kernel averageHistKernel(program, "CalculateAverageHistogram");
    cl::Kernel histogramKernel(program, "CalculateHistogram");
    cl::Kernel varianceHistKernel(program, "CalculateVarianceHistogram");
    
    
    // Create Input Buffers
    cl::Buffer y_pixelBuffer(context, CL_MEM_READ_ONLY, y_Size * sizeof(int), NULL, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Create y_pixelBuffer ERROR: " << err << std::endl;
    }
    cl::Buffer u_pixelBuffer(context, CL_MEM_READ_ONLY, u_Size * sizeof(int), NULL, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Create u_pixelBuffer ERROR: " << err << std::endl;
    }
    cl::Buffer v_pixelBuffer(context, CL_MEM_READ_ONLY, v_Size * sizeof(int), NULL, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Create v_pixelBuffer ERROR: " << err << std::endl;
    }
    cl::Buffer numOfBinsBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &err);
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Create numOfBinsBuffer ERROR: " << err << std::endl;
    }

    // Write Input Buffers
    err = commandQueue.enqueueWriteBuffer(y_pixelBuffer, CL_TRUE, 0, y_Size * sizeof(int), &imageVector[0], NULL, NULL);            
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Write y_pixelBuffer ERROR: " << err << std::endl;
    }
    err = commandQueue.enqueueWriteBuffer(u_pixelBuffer, CL_TRUE, 0, u_Size * sizeof(int), &imageVector[y_Size], NULL, NULL);            
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Write u_pixelBuffer ERROR: " << err << std::endl;
    }
    err = commandQueue.enqueueWriteBuffer(v_pixelBuffer, CL_TRUE, 0, v_Size * sizeof(int), &imageVector[y_Size + u_Size], NULL, NULL);            
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Write v_pixelBuffer ERROR: " << err << std::endl;
    }
    err = commandQueue.enqueueWriteBuffer(numOfBinsBuffer, CL_TRUE, 0, 1 * sizeof(int), &NUM_OF_BINS, NULL, NULL);            
    if (DEBUG_MODE_GPU && err < 0) {
        std::cout << "Write numOfBinsBuffer ERROR: " << err << std::endl;
    }

    // Create Event and Ranges
    cl::Event event;
    cl::NDRange y_GlobalRange(IMG_WIDTH, IMG_HEIGHT);
    cl::NDRange y_LocalRange(y_BlockWidth, y_BlockHeight);
    cl::NDRange u_GlobalRange(IMG_WIDTH/2, IMG_HEIGHT/2);
    cl::NDRange u_LocalRange(u_BlockWidth, u_BlockHeight);
    cl::NDRange v_GlobalRange(IMG_WIDTH/2, IMG_HEIGHT/2);
    cl::NDRange v_LocalRange(v_BlockWidth, v_BlockHeight);

    if (AVERAGE_TEST) {
        std::cout << "\n=========================AVERAGE TEST=========================\n\n";
        // Create Output Buffers
        cl::Buffer y_AverageBuffer(context, CL_MEM_WRITE_ONLY, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_AverageBuffer(context, CL_MEM_WRITE_ONLY, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_AverageBuffer(context, CL_MEM_WRITE_ONLY, v_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<double> y_AverageGPU(y_NumOfBlocks);
        std::vector<double> u_AverageGPU(u_NumOfBlocks);
        std::vector<double> v_AverageGPU(v_NumOfBlocks);
 
        // Create Timer Variables
        double y_ElapsedTimeAverageGPU, u_ElapsedTimeAverageGPU, v_ElapsedTimeAverageGPU;

        // Execute for Channel Y
        averageKernel.setArg(0, y_pixelBuffer);
        averageKernel.setArg(1, y_AverageBuffer);
        err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate for other Channels
        if (ALL_CHANNELS) {
            // Execute for Channel U
            averageKernel.setArg(0, u_pixelBuffer);
            averageKernel.setArg(1, u_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    
            // Execute for Channel V
            averageKernel.setArg(0, v_pixelBuffer);
            averageKernel.setArg(1, v_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_AverageBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_AverageBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_AverageBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageBuffer ERROR: " << err << std::endl;
        }

        // Validate Average of Channel Y
        std::cout << "Validating Y Average GPU: ";
        if (ValidateVector<double>(y_AverageGPU, y_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        if (ALL_CHANNELS) {
            // Validate Average of Channel U
            std::cout << "Validating U Average GPU: ";
            if (ValidateVector<double>(u_AverageGPU, u_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }

            // Validate Average of Channel V
            std::cout << "Validating V Average GPU: ";
            if (ValidateVector<double>(v_AverageGPU, v_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }
        }

        // Generate Output image
        int outputWidth = IMG_WIDTH/BLOCK_WIDTH;
        int outputHeight = IMG_HEIGHT/BLOCK_HEIGHT;

        if (GENERATE_AVERAGE_IMG) {
            std::string fileNameAverage = "Output Images/OUTPUT_IMG_" + std::to_string(outputWidth) + "x" + std::to_string(outputHeight) + "_AVERAGE.yuv";
            std::cout << "Generating average image " << fileNameAverage << ": ";
            std::ofstream outputYUV (fileNameAverage, std::ios::binary);
            if (ALL_CHANNELS) {
                for (int i = 0; i < y_AverageGPU.size(); i++) {
                    outputYUV.put(y_AverageGPU[i]);
                }
                for (int i = 0; i < u_AverageGPU.size(); i++) {
                    outputYUV.put(u_AverageGPU[i]);
                }
                for (int i = 0; i < v_AverageGPU.size(); i++) {
                    outputYUV.put(v_AverageGPU[i]);
                }
            }
            else {
                for (int i = 0; i < y_AverageGPU.size(); i++) {
                    outputYUV.put(y_AverageGPU[i]);
                }
                for (int i = y_Size; i < imageSize; i=i+(BLOCK_WIDTH * BLOCK_HEIGHT)) {
                    outputYUV.put(imageVector[i]);
                }
            }
            outputYUV.close();
            std::cout << "DONE" << std::endl;
        }

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Channel Y (ms) = " << y_ElapsedTimeAverageGPU << std::endl;
        if (ALL_CHANNELS) {
            std::cout << "Elapsed time Channel U (ms) = " << u_ElapsedTimeAverageGPU << std::endl;
            std::cout << "Elapsed time Channel V (ms) = " << v_ElapsedTimeAverageGPU << std::endl;
            std::cout << "Elapsed time Average (Y + U + V) (ms) = " << y_ElapsedTimeAverageGPU + u_ElapsedTimeAverageGPU + v_ElapsedTimeAverageGPU << std::endl;
        }
    }

    if (VARIANCE_TEST_1) {
        std::cout << "\n=======================VARIANCE TEST 1========================\n\n";
        // Create Output Buffers
        cl::Buffer y_AverageBuffer(context, CL_MEM_WRITE_ONLY, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer y_VarianceBuffer(context, CL_MEM_WRITE_ONLY, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_VarianceBuffer ERROR: " << err << std::endl;
        }
 
        cl::Buffer u_AverageBuffer(context, CL_MEM_WRITE_ONLY, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_VarianceBuffer(context, CL_MEM_WRITE_ONLY, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_VarianceBuffer ERROR: " << err << std::endl;
        }

        cl::Buffer v_AverageBuffer(context, CL_MEM_WRITE_ONLY, v_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_VarianceBuffer(context, CL_MEM_WRITE_ONLY, v_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_VarianceBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<double> y_AverageGPU(y_NumOfBlocks);
        std::vector<double> y_VarianceGPU(y_NumOfBlocks);
        std::vector<double> u_AverageGPU(y_NumOfBlocks);
        std::vector<double> u_VarianceGPU(y_NumOfBlocks);
        std::vector<double> v_AverageGPU(y_NumOfBlocks);
        std::vector<double> v_VarianceGPU(y_NumOfBlocks);

        // Create Timer Variables
        double y_ElapsedTimeVarianceGPU, u_ElapsedTimeVarianceGPU, v_ElapsedTimeVarianceGPU;

        // Execute for Channel Y
        averageVarianceKernel.setArg(0, y_pixelBuffer);
        averageVarianceKernel.setArg(1, y_AverageBuffer);
        averageVarianceKernel.setArg(2, y_VarianceBuffer);
        err = commandQueue.enqueueNDRangeKernel(averageVarianceKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y ERROR: " << err << std::endl;
        }
        y_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate for other Channels
        if (ALL_CHANNELS) {
            // Execute for Channel U
            averageVarianceKernel.setArg(0, u_pixelBuffer);
            averageVarianceKernel.setArg(1, u_AverageBuffer);
            averageVarianceKernel.setArg(2, u_VarianceBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageVarianceKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U ERROR: " << err << std::endl;
            }
            u_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    
            // Execute for Channel V
            averageVarianceKernel.setArg(0, v_pixelBuffer);
            averageVarianceKernel.setArg(1, v_AverageBuffer);
            averageVarianceKernel.setArg(2, v_VarianceBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageVarianceKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V ERROR: " << err << std::endl;
            }
            v_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_AverageBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(y_VarianceBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_VarianceBuffer ERROR: " << err << std::endl;
        }

        err = commandQueue.enqueueReadBuffer(u_AverageBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_VarianceBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_VarianceBuffer ERROR: " << err << std::endl;
        }

        err = commandQueue.enqueueReadBuffer(v_AverageBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_VarianceBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_VarianceBuffer ERROR: " << err << std::endl;
        }

        // Validate Average of Channel Y
        std::cout << "Validating Y Average GPU: ";
        if (ValidateVector<double>(y_AverageGPU, y_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        if (ALL_CHANNELS) {
            // Validate Average of Channel U
            std::cout << "Validating U Average GPU: ";
            if (ValidateVector<double>(u_AverageGPU, u_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }

            // Validate Average of Channel V
            std::cout << "Validating V Average GPU: ";
            if (ValidateVector<double>(v_AverageGPU, v_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }
        }

        // Validate Variance of Channel Y
        std::cout << "Validating Y Variance GPU: ";
        if (ValidateVector<double>(y_VarianceGPU, y_VarianceCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        if (ALL_CHANNELS) {
            // Validate Variance of Channel U
            std::cout << "Validating U Variance GPU: ";
            if (ValidateVector<double>(u_VarianceGPU, u_VarianceCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }

            // Validate Variance of Channel V
            std::cout << "Validating V Variance GPU: ";
            if (ValidateVector<double>(v_VarianceGPU, v_VarianceCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }
        }

        // Generate Output image
        int outputWidth = IMG_WIDTH/BLOCK_WIDTH;
        int outputHeight = IMG_HEIGHT/BLOCK_HEIGHT;

        if (GENERATE_VARIANCE_IMG) {
            std::string fileNameVariance = "Output Images/OUTPUT_IMG_" + std::to_string(outputWidth) + "x" + std::to_string(outputHeight) + "_VARIANCE.yuv";
            std::cout << "Generating variance image " << fileNameVariance << ": ";
            std::ofstream outputYUV (fileNameVariance, std::ios::binary);
            if (ALL_CHANNELS) {
                for (int i = 0; i < y_VarianceGPU.size(); i++) {
                    outputYUV.put(y_VarianceGPU[i]);
                }
                for (int i = 0; i < u_VarianceGPU.size(); i++) {
                    outputYUV.put(u_VarianceGPU[i]);
                }
                for (int i = 0; i < v_VarianceGPU.size(); i++) {
                    outputYUV.put(v_VarianceGPU[i]);
                }
            }
            else {
                for (int i = 0; i < y_VarianceGPU.size(); i++) {
                    outputYUV.put(y_VarianceGPU.at(i));
                }
                for (int i = y_Size; i < imageSize; i=i+(BLOCK_WIDTH * BLOCK_HEIGHT)) {
                    outputYUV.put(imageVector.at(0));
                }
            }
            outputYUV.close();
            std::cout << "DONE" << std::endl;
        }

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
         std::cout << "Elapsed time Channel Y (ms) = " << y_ElapsedTimeVarianceGPU << std::endl;
        if (ALL_CHANNELS) {
            std::cout << "Elapsed time Channel U (ms) = " << u_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Channel V (ms) = " << v_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Average + Variance (Y + U + V) (ms) = " << y_ElapsedTimeVarianceGPU + u_ElapsedTimeVarianceGPU + v_ElapsedTimeVarianceGPU << std::endl;
        }
    }

    if (VARIANCE_TEST_2) {
        std::cout << "\n=======================VARIANCE TEST 2========================\n\n";
        // Create Output Buffers
        cl::Buffer y_AverageBuffer(context, CL_MEM_WRITE_ONLY, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer y_VarianceBuffer(context, CL_MEM_WRITE_ONLY, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_VarianceBuffer ERROR: " << err << std::endl;
        }
 
        cl::Buffer u_AverageBuffer(context, CL_MEM_WRITE_ONLY, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_VarianceBuffer(context, CL_MEM_WRITE_ONLY, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_VarianceBuffer ERROR: " << err << std::endl;
        }

        cl::Buffer v_AverageBuffer(context, CL_MEM_WRITE_ONLY, v_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_VarianceBuffer(context, CL_MEM_WRITE_ONLY, v_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_VarianceBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<double> y_AverageGPU(y_NumOfBlocks);
        std::vector<double> y_VarianceGPU(y_NumOfBlocks);
        std::vector<double> u_AverageGPU(y_NumOfBlocks);
        std::vector<double> u_VarianceGPU(y_NumOfBlocks);
        std::vector<double> v_AverageGPU(y_NumOfBlocks);
        std::vector<double> v_VarianceGPU(y_NumOfBlocks);

        // Create Timer Variables
        double y_ElapsedTimeAverageGPU, u_ElapsedTimeAverageGPU, v_ElapsedTimeAverageGPU;
        double y_ElapsedTimeVarianceGPU, u_ElapsedTimeVarianceGPU, v_ElapsedTimeVarianceGPU;

        // Execute Average for Channel Y
        averageKernel.setArg(0, y_pixelBuffer);
        averageKernel.setArg(1, y_AverageBuffer);
        err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y Average ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

        // Execute Variance for Channel Y
        varianceKernel.setArg(0, y_pixelBuffer);
        varianceKernel.setArg(1, y_AverageBuffer);
        varianceKernel.setArg(2, y_VarianceBuffer);
        err = commandQueue.enqueueNDRangeKernel(varianceKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y Variance ERROR: " << err << std::endl;
        }
        y_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate for other Channels
        if (ALL_CHANNELS) {
            // Execute Average for Channel U
            averageKernel.setArg(0, u_pixelBuffer);
            averageKernel.setArg(1, u_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U Average ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

            // Execute Variance for Channel U
            varianceKernel.setArg(0, u_pixelBuffer);
            varianceKernel.setArg(1, u_AverageBuffer);
            varianceKernel.setArg(2, u_VarianceBuffer);
            err = commandQueue.enqueueNDRangeKernel(varianceKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U Variance ERROR: " << err << std::endl;
            }
            u_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
            // Execute Average for Channel V
            averageKernel.setArg(0, v_pixelBuffer);
            averageKernel.setArg(1, v_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V Average ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

            // Execute Variance for Channel V
            varianceKernel.setArg(0, v_pixelBuffer);
            varianceKernel.setArg(1, v_AverageBuffer);
            varianceKernel.setArg(2, v_VarianceBuffer);
            err = commandQueue.enqueueNDRangeKernel(varianceKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U Variance ERROR: " << err << std::endl;
            }
            v_ElapsedTimeVarianceGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_AverageBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(y_VarianceBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_VarianceBuffer ERROR: " << err << std::endl;
        }

        err = commandQueue.enqueueReadBuffer(u_AverageBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_VarianceBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_VarianceBuffer ERROR: " << err << std::endl;
        }

        err = commandQueue.enqueueReadBuffer(v_AverageBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_VarianceBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_VarianceGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_VarianceBuffer ERROR: " << err << std::endl;
        }

        // Validate Average of Channel Y
        std::cout << "Validating Y Average GPU: ";
        if (ValidateVector<double>(y_AverageGPU, y_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        if (ALL_CHANNELS) {
            // Validate Average of Channel U
            std::cout << "Validating U Average GPU: ";
            if (ValidateVector<double>(u_AverageGPU, u_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }

            // Validate Average of Channel V
            std::cout << "Validating V Average GPU: ";
            if (ValidateVector<double>(v_AverageGPU, v_AverageCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }
        }

        // Validate Variance of Channel Y
        std::cout << "Validating Y Variance GPU: ";
        if (ValidateVector<double>(y_VarianceGPU, y_VarianceCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        if (ALL_CHANNELS) {
            // Validate Variance of Channel U
            std::cout << "Validating U Variance GPU: ";
            if (ValidateVector<double>(u_VarianceGPU, u_VarianceCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }

            // Validate Variance of Channel V
            std::cout << "Validating V Variance GPU: ";
            if (ValidateVector<double>(v_VarianceGPU, v_VarianceCPU)) {
                std::cout << "PASS" << std::endl;
            }
            else {
                std::cout << "FAIL" << std::endl;
            }
        }

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Channel Y Average (ms) = " << y_ElapsedTimeAverageGPU << std::endl;
        std::cout << "Elapsed time Channel Y Variance (ms) = " << y_ElapsedTimeVarianceGPU << std::endl;
        std::cout << "Elapsed time Channel Y (Avg + Var) (ms) = " << y_ElapsedTimeAverageGPU + y_ElapsedTimeVarianceGPU << std::endl;
        if (ALL_CHANNELS) {
            std::cout << "Elapsed time Channel U Average (ms) = " << u_ElapsedTimeAverageGPU << std::endl;
            std::cout << "Elapsed time Channel U Variance (ms) = " << u_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Channel U (Avg + Var) (ms) = " << u_ElapsedTimeAverageGPU + u_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Channel V Average (ms) = " << v_ElapsedTimeAverageGPU << std::endl;
            std::cout << "Elapsed time Channel V Variance (ms) = " << v_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Channel V (Avg + Var) (ms) = " << v_ElapsedTimeAverageGPU + v_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time Average (Y + U + V) (ms) = " << y_ElapsedTimeAverageGPU + u_ElapsedTimeAverageGPU + v_ElapsedTimeAverageGPU << std::endl;
            std::cout << "Elapsed time Variance (Y + U + V) (ms) = " << y_ElapsedTimeVarianceGPU + u_ElapsedTimeVarianceGPU + v_ElapsedTimeVarianceGPU << std::endl;
            std::cout << "Elapsed time (Avg + Var) (Y + U + V) (ms) = " << y_ElapsedTimeAverageGPU + u_ElapsedTimeAverageGPU + v_ElapsedTimeAverageGPU + y_ElapsedTimeVarianceGPU + u_ElapsedTimeVarianceGPU + v_ElapsedTimeVarianceGPU << std::endl;
        }
    }

    if (AVERAGE_HISTOGRAM_TEST_1) {
        std::cout << "\n==================AVERAGE HISTOGRAM TEST 1====================\n\n";
        // Create Output Buffers
        cl::Buffer y_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageHistBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<int> y_AverageBinsGPU(NUM_OF_BINS);
        std::vector<int> u_AverageBinsGPU(NUM_OF_BINS);
        std::vector<int> v_AverageBinsGPU(NUM_OF_BINS);

        // Initialize Hist Buffer
        err = commandQueue.enqueueWriteBuffer(y_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(u_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(v_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageHistBuffer ERROR: " << err << std::endl;
        }
        
        // Create Timer Variables
        double y_ElapsedTimeAverageHistGPU;
        double u_ElapsedTimeAverageHistGPU;
        double v_ElapsedTimeAverageHistGPU;

        // Execute Histogram for Channel Y
        averageHistKernel.setArg(0, y_pixelBuffer);
        averageHistKernel.setArg(1, numOfBinsBuffer);
        averageHistKernel.setArg(2, y_AverageHistBuffer);
        err = commandQueue.enqueueNDRangeKernel(averageHistKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate other channels
        if (ALL_CHANNELS) {
            // U Channel
            averageHistKernel.setArg(0, u_pixelBuffer);
            averageHistKernel.setArg(1, numOfBinsBuffer);
            averageHistKernel.setArg(2, u_AverageHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageHistKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
            // V Channel
            averageHistKernel.setArg(0, v_pixelBuffer);
            averageHistKernel.setArg(1, numOfBinsBuffer);
            averageHistKernel.setArg(2, v_AverageHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageHistKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageHistBuffer ERROR: " << err << std::endl;
        }
        
        // Validate Average Histogram of Channel Y
        std::cout << "Validating Y Average Histogram GPU: ";
        if (ValidateVector<int>(y_AverageBinsGPU, y_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }
        std::cout << "Validating U Average Histogram GPU: ";
        if (ValidateVector<int>(u_AverageBinsGPU, u_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        } 
        std::cout << "Validating V Average Histogram GPU: ";
        if (ValidateVector<int>(v_AverageBinsGPU, v_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }  

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Channel Y (ms) = " << y_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel U (ms) = " << u_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel V (ms) = " << v_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time (Y + U + V) (ms) = " << y_ElapsedTimeAverageHistGPU + u_ElapsedTimeAverageHistGPU + v_ElapsedTimeAverageHistGPU << std::endl;

        // Export Histogram
        if (EXPORT_AVERAGE_HISTOGRAM) {
            std::ofstream outputHistogramGPU ("Output Histograms/AVERAGE_HISTOGRAM_Y.txt");
            for (int i = 0; i < y_AverageBinsGPU.size(); i++) {
                outputHistogramGPU << y_AverageBinsGPU[i];
                if (i + 1 < y_AverageBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();

            outputHistogramGPU = std::ofstream("Output Histograms/AVERAGE_HISTOGRAM_U.txt");
            for (int i = 0; i < u_AverageBinsGPU.size(); i++) {
                outputHistogramGPU << u_AverageBinsGPU[i];
                if (i + 1 < u_AverageBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();

            outputHistogramGPU = std::ofstream("Output Histograms/AVERAGE_HISTOGRAM_V.txt");
            for (int i = 0; i < v_AverageBinsGPU.size(); i++) {
                outputHistogramGPU << v_AverageBinsGPU[i];
                if (i + 1 < v_AverageBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();
        }
    }

    if (AVERAGE_HISTOGRAM_TEST_2) {
        std::cout << "\n==================AVERAGE HISTOGRAM TEST 2====================\n\n";
        // Create Output Buffers
        cl::Buffer y_AverageBuffer(context, CL_MEM_READ_WRITE, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer y_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_AverageBuffer(context, CL_MEM_READ_WRITE, u_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_AverageBuffer(context, CL_MEM_READ_WRITE, y_NumOfBlocks * sizeof(double), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_AverageHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageHistBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<double> y_AverageGPU(y_NumOfBlocks);
        std::vector<int> y_AverageBinsGPU(NUM_OF_BINS);
        std::vector<double> u_AverageGPU(u_NumOfBlocks);
        std::vector<int> u_AverageBinsGPU(NUM_OF_BINS);
        std::vector<double> v_AverageGPU(v_NumOfBlocks);
        std::vector<int> v_AverageBinsGPU(NUM_OF_BINS);

        // Initialize Hist Buffer
        err = commandQueue.enqueueWriteBuffer(y_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(u_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(v_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageHistBuffer ERROR: " << err << std::endl;
        }

        
        // Create Timer Variables
        double y_ElapsedTimeAverageGPU;
        double y_ElapsedTimeAverageHistGPU;
        double u_ElapsedTimeAverageGPU;
        double u_ElapsedTimeAverageHistGPU;
        double v_ElapsedTimeAverageGPU;
        double v_ElapsedTimeAverageHistGPU;

        // Execute Average for Channel Y
        averageKernel.setArg(0, y_pixelBuffer);
        averageKernel.setArg(1, y_AverageBuffer);
        err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y Average ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

        // Execute Histogram for Channel Y
        histogramKernel.setArg(0, y_AverageBuffer);
        histogramKernel.setArg(1, numOfBinsBuffer);
        histogramKernel.setArg(2, y_AverageHistBuffer);
        err = commandQueue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(y_NumOfBlocks), 1, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y HIST ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate other channels
        if (ALL_CHANNELS) {
            // U Channel
            averageKernel.setArg(0, u_pixelBuffer);
            averageKernel.setArg(1, u_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U Average ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

            histogramKernel.setArg(0, u_AverageBuffer);
            histogramKernel.setArg(1, numOfBinsBuffer);
            histogramKernel.setArg(2, u_AverageHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(u_NumOfBlocks), 1, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U HIST ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
            // V Channel
            averageKernel.setArg(0, v_pixelBuffer);
            averageKernel.setArg(1, v_AverageBuffer);
            err = commandQueue.enqueueNDRangeKernel(averageKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V Average ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

            histogramKernel.setArg(0, v_AverageBuffer);
            histogramKernel.setArg(1, numOfBinsBuffer);
            histogramKernel.setArg(2, v_AverageHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(v_NumOfBlocks), 1, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V HIST ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_AverageBuffer, CL_TRUE, 0, y_NumOfBlocks * sizeof(double), &y_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(y_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_AverageBuffer, CL_TRUE, 0, u_NumOfBlocks * sizeof(double), &u_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_AverageHistBuffer ERROR: " << err << std::endl;
        }
                err = commandQueue.enqueueReadBuffer(v_AverageBuffer, CL_TRUE, 0, v_NumOfBlocks * sizeof(double), &v_AverageGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_AverageHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_AverageBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_AverageHistBuffer ERROR: " << err << std::endl;
        }
        
        // Validate
        std::cout << "Validating Y Average GPU: ";
        if (ValidateVector<double>(y_AverageGPU, y_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        std::cout << "Validating Y Average Histogram GPU: ";
        if (ValidateVector<int>(y_AverageBinsGPU, y_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        std::cout << "Validating U Average GPU: ";
        if (ValidateVector<double>(u_AverageGPU, u_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        std::cout << "Validating U Average Histogram GPU: ";
        if (ValidateVector<int>(u_AverageBinsGPU, u_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        } 

        std::cout << "Validating V Average GPU: ";
        if (ValidateVector<double>(v_AverageGPU, v_AverageCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

        std::cout << "Validating V Average Histogram GPU: ";
        if (ValidateVector<int>(v_AverageBinsGPU, v_AverageBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        } 

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Channel Y Average (ms) = " << y_ElapsedTimeAverageGPU << std::endl;
        std::cout << "Elapsed time Channel Y Histogram (ms) = " << y_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel Y Avg + Hist (ms) = " << y_ElapsedTimeAverageGPU + y_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel U Average (ms) = " << u_ElapsedTimeAverageGPU << std::endl;
        std::cout << "Elapsed time Channel U Histogram (ms) = " << u_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel U Avg + Hist (ms) = " << u_ElapsedTimeAverageGPU + u_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel V Average (ms) = " << v_ElapsedTimeAverageGPU << std::endl;
        std::cout << "Elapsed time Channel V Histogram (ms) = " << v_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time Channel V Avg + Hist (ms) = " << v_ElapsedTimeAverageGPU + v_ElapsedTimeAverageHistGPU << std::endl;
        std::cout << "Elapsed time (Y + U + V) Avg + Hist (ms) = " << y_ElapsedTimeAverageGPU + y_ElapsedTimeAverageHistGPU + u_ElapsedTimeAverageGPU + u_ElapsedTimeAverageHistGPU + v_ElapsedTimeAverageGPU + v_ElapsedTimeAverageHistGPU << std::endl;
    }

    if (VARIANCE_HISTOGRAM_TEST_1) {
        std::cout << "\n==================VARIANCE HISTOGRAM TEST 1====================\n\n";
        // Create Output Buffers
        cl::Buffer y_VarianceHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create y_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer u_VarianceHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create u_AverageHistBuffer ERROR: " << err << std::endl;
        }
        cl::Buffer v_VarianceHistBuffer(context, CL_MEM_READ_WRITE, NUM_OF_BINS * sizeof(int), NULL, &err);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Create v_AverageHistBuffer ERROR: " << err << std::endl;
        }

        // Create Output Vectors
        std::vector<int> y_VarianceBinsGPU(NUM_OF_BINS);
        std::vector<int> u_VarianceBinsGPU(NUM_OF_BINS);
        std::vector<int> v_VarianceBinsGPU(NUM_OF_BINS);

        // Initialize Hist Buffer
        err = commandQueue.enqueueWriteBuffer(y_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(u_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueWriteBuffer(v_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        
        // Create Timer Variables
        double y_ElapsedTimeAverageHistGPU;
        double u_ElapsedTimeAverageHistGPU;
        double v_ElapsedTimeAverageHistGPU;

        // Execute Histogram for Channel Y
        varianceHistKernel.setArg(0, y_pixelBuffer);
        varianceHistKernel.setArg(1, numOfBinsBuffer);
        varianceHistKernel.setArg(2, y_VarianceHistBuffer);
        err = commandQueue.enqueueNDRangeKernel(varianceHistKernel, cl::NullRange, y_GlobalRange, y_LocalRange, NULL, &event);
        event.wait();
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Execution Y ERROR: " << err << std::endl;
        }
        y_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        
        // Calculate other channels
        if (ALL_CHANNELS) {
            // U Channel
            varianceHistKernel.setArg(0, u_pixelBuffer);
            varianceHistKernel.setArg(1, numOfBinsBuffer);
            varianceHistKernel.setArg(2, u_VarianceHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(varianceHistKernel, cl::NullRange, u_GlobalRange, u_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution U ERROR: " << err << std::endl;
            }
            u_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
            // V Channel
            varianceHistKernel.setArg(0, v_pixelBuffer);
            varianceHistKernel.setArg(1, numOfBinsBuffer);
            varianceHistKernel.setArg(2, v_VarianceHistBuffer);
            err = commandQueue.enqueueNDRangeKernel(varianceHistKernel, cl::NullRange, v_GlobalRange, v_LocalRange, NULL, &event);
            event.wait();
            if (DEBUG_MODE_GPU && err < 0) {
                std::cout << "Execution V ERROR: " << err << std::endl;
            }
            v_ElapsedTimeAverageHistGPU = (1e-6) * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
            
        }

        // Read responsess
        err = commandQueue.enqueueReadBuffer(y_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &y_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading y_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(u_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &u_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading u_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        err = commandQueue.enqueueReadBuffer(v_VarianceHistBuffer, CL_TRUE, 0, NUM_OF_BINS * sizeof(int), &v_VarianceBinsGPU[0], NULL, NULL);
        if (DEBUG_MODE_GPU && err < 0) {
            std::cout << "Reading v_VarianceHistBuffer ERROR: " << err << std::endl;
        }
        
        // Validate Average Histogram of Channel Y
        std::cout << "Validating Y Variance Histogram GPU: ";
        if (ValidateVector<int>(y_VarianceBinsGPU, y_VarianceBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }
        std::cout << "Validating U Variance Histogram GPU: ";
        if (ValidateVector<int>(u_VarianceBinsGPU, u_VarianceBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        } 
        std::cout << "Validating V Variance Histogram GPU: ";
        if (ValidateVector<int>(v_VarianceBinsGPU, v_VarianceBinsCPU)) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }  

        std::cout << "\n---------------------------SUMMARY----------------------------\n\n";
        std::cout << "Elapsed time Channel Y (ms) = " << y_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel U (ms) = " << u_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time Channel V (ms) = " << v_ElapsedTimeVarianceHistCPU << std::endl;
        std::cout << "Elapsed time (Y + U + V) (ms) = " << y_ElapsedTimeVarianceHistCPU + u_ElapsedTimeVarianceHistCPU + v_ElapsedTimeVarianceHistCPU << std::endl;

        // Export Histogram
        if (EXPORT_VARIANCE_HISTOGRAM) {
            std::ofstream outputHistogramGPU ("Output Histograms/VARIANCE_HISTOGRAM_Y.txt");
            for (int i = 0; i < y_VarianceBinsGPU.size(); i++) {
                outputHistogramGPU << y_VarianceBinsGPU[i];
                if (i + 1 < y_VarianceBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();

            outputHistogramGPU = std::ofstream("Output Histograms/VARIANCE_HISTOGRAM_U.txt");
            for (int i = 0; i < u_VarianceBinsGPU.size(); i++) {
                outputHistogramGPU << u_VarianceBinsGPU[i];
                if (i + 1 < u_VarianceBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();

            outputHistogramGPU = std::ofstream("Output Histograms/VARIANCE_HISTOGRAM_V.txt");
            for (int i = 0; i < v_VarianceBinsGPU.size(); i++) {
                outputHistogramGPU << v_VarianceBinsGPU[i];
                if (i + 1 < v_VarianceBinsGPU.size()) {
                    outputHistogramGPU << "\n";
                }
            }
            outputHistogramGPU.close();
        }
    }

    imageVector.clear();

    return 0;
}
