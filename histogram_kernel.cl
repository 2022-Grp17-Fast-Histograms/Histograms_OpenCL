#pragma CL_VERSION_3_0

//inline PTX atomic_add using float
void atomic_add_float(global float *p, float val)
{
    float prev;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;" 
        : "=f"(prev) 
        : "l"(p) , "f"(val) 
        : "memory" 
    );
}

kernel void calculateAverage(global const int *pixels, global float *average, local int *blockSum) {
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
        
        average[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = (float)blockSum[0]/blockSize;

    }
}

kernel void calculateMovingAverage(global const int *pixels, global float *average, local float *blockAvg) {
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

kernel void calculateVariance(global const int *pixels, global const float *average, global float *variance, local float *blockSum) {
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
        variance[block_linear] = (float)blockSum[0]/blockSize;
    }
}

kernel void calculateAverageAndVariance(global const int *pixels, global float *average, global float *variance, local int *blockSumAverage, local float *blockSumVariance) {
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
        average[block_linear] = (float)blockSumAverage[0]/blockSize;
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
        variance[block_linear] = (float)blockSumVariance[0]/blockSize;
    }
}

kernel void calculateOnePassVariance(global const int *pixels, global float *average, global float *variance, local int *blockSumAverage, local int *blockSumSquares) {
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
        average[block_linear] = (float)blockSumAverage[0]/blockSize;
        variance[block_linear] = (float)blockSumSquares[0]/blockSize - average[block_linear] * average[block_linear];
    }
}

kernel void calculateAverageHistogram(global const int *pixels, global const int *numOfBins, global int *bins, local int *blockSum) {
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
        int average = blockSum[0]/blockSize;
        
        // Calculate bin
        int interval = (average*numOfBins[0])>>8;
        atomic_inc(&bins[interval]);
    }    
}

kernel void calculateVarianceHistogram(global const int *pixels, global const int *numOfBins, global int *bins, local int *blockSumAverage, local int *blockSumVariance, local int *average) {
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
        average[0] = blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average[0]) * (pixels[g_linear] - average[0]);

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
        int variance = blockSumVariance[0]/blockSize;

        // Calculate bin
        int interval = (average[0]*numOfBins[0])>>8;
        atomic_add(&bins[interval], variance);
    }
}

kernel void calculateOnePassVarianceHistogram(global const int *pixels, global const int *numOfBins, global float *bins, local int *blockSumAverage, local int *blockSumSquares) {
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
        float average = (float)blockSumAverage[0]/blockSize;

        // Calculate variance
        float variance = (float)blockSumSquares[0]/blockSize - average * average;

        // Calculate bin
        int interval = ((int)average*numOfBins[0])>>8;
        atomic_add_float(&bins[interval], variance);
    }
}

kernel void calculateAllHistograms(global const int *pixels, global const int *numOfBins, global int *averageBins, global int *varianceBins, local int *blockSumAverage, local int *blockSumVariance, local int *average) { 
    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

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
        average[0] = blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average[0]) * (pixels[g_linear] - average[0]);

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
        int variance = blockSumVariance[0]/blockSize;

        // Calculate bin
        int interval = (average[0]*numOfBins[0])>>8;
        atomic_inc(&averageBins[interval]);
        atomic_add(&varianceBins[interval], variance);
    }
}

kernel void calculateAllHistogramsOnePass(global const int *pixels, global const int *numOfBins, global int *averageBins, global float *varianceBins, local int *blockSumAverage, local int *blockSumSquares) {
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
        float average = (float)blockSumAverage[0]/blockSize;

        // Calculate variance
        float variance = (float)blockSumSquares[0]/blockSize - average * average;

        // Calculate bin
        int interval = ((int)average*numOfBins[0])>>8;

        atomic_inc(&averageBins[interval]);
        atomic_add_float(&varianceBins[interval], variance);
    }
}

kernel void calculateHistogram(global const float *input, global const int *numOfBins, global int *bins) {
    // Get Blocal Id
    int i = get_global_id(0);

    // Calculate bin
    int interval = ((int)input[i]*numOfBins[0])>>8;
    atomic_inc(&bins[interval]);
}