#pragma CL_VERSION_3_0

__kernel void CalculateAverage(__global const int *pixels, __global double *average) {
    __local int blockSum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy from global memory to local memory
    blockSum[l_linear] = pixels[g_linear];

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear < stride) {
            blockSum[l_linear] += blockSum[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate block linear position
        average[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = (double)blockSum[0]/blockSize;
    }
}

__kernel void CalculateMovingAverage(__global const int *pixels, __global double *average) {
    __local double blockAvg[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy from global memory to local memory
    blockAvg[l_linear] = pixels[g_linear];
    
    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear < stride) {
            blockAvg[l_linear] += (blockAvg[l_linear + stride] - blockAvg[l_linear])/(l_linear+1);
        }
    }
    
    // Update average array
    if (l_linear == 0) {
        // Calculate block linear position
        average[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = blockAvg[0];
    }
}

__kernel void CalculateVariance(__global const int *pixels, __global const double *average, __global double *variance) {
    __local double blockSum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Calculate initial portion of variance and copy to local memory
    blockSum[l_linear] = (pixels[g_linear] - average[block_linear]) * (pixels[g_linear] - average[block_linear]);
    
    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSum[l_linear] += blockSum[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        variance[block_linear] = (double)blockSum[0]/blockSize;
    }
}

__kernel void CalculateAverageAndVariance(__global const int *pixels, __global double *average, __global double *variance) {
    __local int blockSumAverage[64];
    __local double blockSumVariance[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    
    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average[block_linear] = (double)blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average[block_linear]) * (pixels[g_linear] - average[block_linear]);

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumVariance[l_linear] += blockSumVariance[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        variance[block_linear] = (double)blockSumVariance[0]/blockSize;
    }
}

__kernel void CalculateOnePassVariance(__global const int *pixels, __global double *average, __global double *variance) {
    __local int blockSumAverage[64];
    __local int blockSumSquares[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    blockSumSquares[l_linear] = pixels[g_linear] * pixels[g_linear];

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average[block_linear] = (double)blockSumAverage[0]/blockSize;
        variance[block_linear] = (double)blockSumSquares[0]/blockSize - average[block_linear] * average[block_linear];
    }
}

__kernel void CalculateAverageHistogram(__global const int *pixels, __global const int *numOfBins, __global int *bins) {
    __local int blockSum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy from global memory to local memory
    blockSum[l_linear] = pixels[g_linear];

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear < stride) {
            blockSum[l_linear] += blockSum[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        double average = (double)blockSum[0]/blockSize;
        
        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_inc(&bins[interval]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }    
}

__kernel void CalculateVarianceHistogram(__global const int *pixels, __global const int *numOfBins, __global atomic_int *bins) {
    __local int blockSumAverage[64];
    __local double average;
    __local double blockSumVariance[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    
    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average = (double)blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average) * (pixels[g_linear] - average);

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumVariance[l_linear] += blockSumVariance[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        double variance = (double)blockSumVariance[0]/blockSize;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_fetch_add_explicit(&bins[interval], variance, memory_order_relaxed, memory_scope_work_group);
    }
}

__kernel void CalculateOnePassVarianceHistogram(__global const int *pixels, __global const int *numOfBins, __global atomic_int *bins) {
    __local int blockSumAverage[64];
    __local int blockSumSquares[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    blockSumSquares[l_linear] = pixels[g_linear] * pixels[g_linear];

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        double average = (double)blockSumAverage[0]/blockSize;

        // Calculate variance
        double variance = (double)blockSumSquares[0]/blockSize - average * average;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_fetch_add_explicit(&bins[interval], variance, memory_order_relaxed, memory_scope_work_group);
    }
}

__kernel void CalculateAllHistograms(__global const int *pixels, __global const int *numOfBins, __global int *averageBins, __global atomic_int *varianceBins) {
    __local int blockSumAverage[64];
    __local double average;
    __local double blockSumVariance[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    
    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average = (double)blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average) * (pixels[g_linear] - average);

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumVariance[l_linear] += blockSumVariance[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        double variance = (double)blockSumVariance[0]/blockSize;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_inc(&averageBins[interval]);
        atomic_fetch_add_explicit(&varianceBins[interval], variance, memory_order_relaxed, memory_scope_work_group);
    }
}

__kernel void CalculateAllHistogramsOnePass(__global const int *pixels, __global const int *numOfBins, __global int *averageBins, __global atomic_int *varianceBins) {
    __local int blockSumAverage[64];
    __local int blockSumSquares[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Get block position and size
    int blockSize = get_local_size(0) * get_local_size(1);

    // Copy memory from global to local
    blockSumAverage[l_linear] = pixels[g_linear];
    blockSumSquares[l_linear] = pixels[g_linear] * pixels[g_linear];

    // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        double average = (double)blockSumAverage[0]/blockSize;

        // Calculate variance
        double variance = (double)blockSumSquares[0]/blockSize - average * average;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);

        atomic_inc(&averageBins[interval]);
        atomic_fetch_add_explicit(&varianceBins[interval], variance, memory_order_relaxed, memory_scope_work_group);
    }
}

__kernel void CalculateHistogram(__global const double *input, __global const int *numOfBins, __global atomic_int *bins) {
    // Get Blocal Id
    int i = get_global_id(0);

    // Get Local Id
    int j = get_local_id(0);

    // Calculate bin
    int interval = input[i]/(256/numOfBins[0]);
    atomic_fetch_add_explicit(&bins[interval], 1, memory_order_relaxed, memory_scope_work_group);
}