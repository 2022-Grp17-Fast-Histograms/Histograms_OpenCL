#pragma CL_VERSION_3_0

//inline PTX atomic_add using float (Only works for nvidia)
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
    /* for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + stride];
        }
    } */

    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize >= 64 && l_linear < 32) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 32];
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 16];
    }

    if (blockSize >= 16 && l_linear < 8) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 8];
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 4];
    }

    if (blockSize >= 4 && l_linear < 2) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 2];
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 1];
    }      

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average[0] = blockSumAverage[0]/blockSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blockSumVariance[l_linear] = (pixels[g_linear] - average[0]) * (pixels[g_linear] - average[0]);

    /* // Loop summing by splitting work group in half
    for (uint stride = blockSize/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blockSumVariance[l_linear] += blockSumVariance[l_linear + stride];
        }
    } */

    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize >= 64 && l_linear < 32) {
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 32];
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 16];
    }

    if (blockSize >= 16 && l_linear < 8) {
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 8];
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 4];
    }

    if (blockSize >= 4 && l_linear < 2) {
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 2];
        blockSumVariance[l_linear] += blockSumVariance[l_linear + 1];
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

    barrier(CLK_LOCAL_MEM_FENCE); 

    if (blockSize == 1024 && l_linear < 512) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 512];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && l_linear < 256) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 256];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && l_linear < 128) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 128];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && l_linear < 64) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 64];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 64];
    }

    //barrier(CLK_LOCAL_MEM_FENCE);

    if (l_linear < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 32];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 32];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 16];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 8];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 8];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 4];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 2];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 2];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 1];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 1];
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

kernel void calculateAllHistogramsOnePassHalf(global const int *pixels, global const int *numOfBins, global int *averageBins, global float *varianceBins, local int *blockSumAverage, local int *blockSumSquares) {
    // Get local id
    int l_linear = get_local_linear_id();

    // Get global size
    int globalWidth = get_global_size(0) * 2;

    // Get block dimensions
    int blockWidth = get_local_size(0);
    int blockHeight = get_local_size(1);
    int blockSize = blockWidth * blockHeight;

    // Get global id
    int g_x = get_group_id(0) * (2*blockWidth) + get_local_id(0);
    int g_y = get_group_id(1) * (2*blockHeight) + get_local_id(1);
    
    int g_linear = (g_y * globalWidth) + g_x;
    int g_linear_offset_x = g_linear + blockWidth;
    int g_linear_offset_y = g_linear + (blockHeight * globalWidth);
    int g_linear_offset_xy = g_linear_offset_y + blockWidth;
    
    // Copy memory from global to local and add offset positions
    blockSumAverage[l_linear] = pixels[g_linear];
    blockSumAverage[l_linear] += pixels[g_linear_offset_x];
    blockSumAverage[l_linear] += pixels[g_linear_offset_y];
    blockSumAverage[l_linear] += pixels[g_linear_offset_xy];

    blockSumSquares[l_linear] = pixels[g_linear] * pixels[g_linear];
    blockSumSquares[l_linear] += pixels[g_linear_offset_x] * pixels[g_linear_offset_x];
    blockSumSquares[l_linear] += pixels[g_linear_offset_y] * pixels[g_linear_offset_y];
    blockSumSquares[l_linear] += pixels[g_linear_offset_xy] * pixels[g_linear_offset_xy];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize == 1024 && l_linear < 512) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 512];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512 && l_linear < 256) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 256];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256 && l_linear < 128) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 128];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128 && l_linear < 64) {
        blockSumAverage[l_linear] += blockSumAverage[l_linear + 64];
        blockSumSquares[l_linear] += blockSumSquares[l_linear + 64];
    }

    if (l_linear < 32) {
        if (blockSize >= 64) {
            barrier(CLK_LOCAL_MEM_FENCE);
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 32];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 32];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 16];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 16];
        }

        if (blockSize >= 16) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 8];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 8];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 4];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 4];
        }

        if (blockSize >= 4) {
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 2];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 2];
            blockSumAverage[l_linear] += blockSumAverage[l_linear + 1];
            blockSumSquares[l_linear] += blockSumSquares[l_linear + 1];
        } 
    }
    
    // Update average array
    if (l_linear == 0) {
        // Calculate average        
        float average = (float)blockSumAverage[0]/(blockSize*4);

        // Calculate variance
        float variance = (float)blockSumSquares[0]/(blockSize*4) - average * average;

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