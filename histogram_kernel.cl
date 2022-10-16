__kernel void CalculateAverage(__global const int *pixels, __global double *average) {
    __local int blocksum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Copy from global memory to local memory
    blocksum[l_linear] = pixels[g_linear];

    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear < stride) {
            blocksum[l_linear] += blocksum[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate block linear position
        average[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = (double)blocksum[0]/block_size;
    }
}

__kernel void CalculateVariance(__global const int *pixels, __global const double *average, __global double *variance) {
    __local double blocksum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Calculate initial portion of variance and copy to local memory
    blocksum[l_linear] = (pixels[g_linear] - average[block_linear]) * (pixels[g_linear] - average[block_linear]);
    
    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blocksum[l_linear] += blocksum[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        variance[block_linear] = (double)blocksum[0]/block_size;
    }
}

__kernel void CalculateAverageAndVariance(__global const int *pixels, __global double *average, __global double *variance) {
    __local int blocksum_average[64];
    __local double blocksum_variance[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Copy memory from global to local
    blocksum_average[l_linear] = pixels[g_linear];

    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);
    
    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blocksum_average[l_linear] += blocksum_average[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average[block_linear] = (double)blocksum_average[0]/block_size;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blocksum_variance[l_linear] = (pixels[g_linear] - average[block_linear]) * (pixels[g_linear] - average[block_linear]);

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blocksum_variance[l_linear] += blocksum_variance[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        variance[block_linear] = (double)blocksum_variance[0]/block_size;
    }
}

__kernel void CalculateAverageHistogram(__global const int *pixels, __global const int *numOfBins, __global int *bins) {
    __local int blocksum[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();

    // Copy from global memory to local memory
    blocksum[l_linear] = pixels[g_linear];

    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear < stride) {
            blocksum[l_linear] += blocksum[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        double average = (double)blocksum[0]/block_size;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_inc(&bins[interval]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }    
}

__kernel void CalculateVarianceHistogram(__global const int *pixels, __global const int *numOfBins, __global atomic_int *bins) {
    __local int blocksum_average[64];
    __local double average;
    __local double blocksum_variance[64];

    // Get local id
    int l_linear = get_local_linear_id();

    // Get global id
    int g_linear = get_global_linear_id();
    
    // Block linear position
    int block_linear = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // Copy memory from global to local
    blocksum_average[l_linear] = pixels[g_linear];

    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);
    
    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blocksum_average[l_linear] += blocksum_average[l_linear + stride];
        }
    }

    // Update average array
    if (l_linear == 0) {
        // Calculate average
        average = (double)blocksum_average[0]/block_size;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate initial portion of variance and copy to local memory
    blocksum_variance[l_linear] = (pixels[g_linear] - average) * (pixels[g_linear] - average);

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (l_linear < stride) {
            blocksum_variance[l_linear] += blocksum_variance[l_linear + stride];
        }
    }

    // Update variance array
    if (l_linear == 0) {
        // Calculate variance
        double variance = (double)blocksum_variance[0]/block_size;

        // Calculate bin
        int interval = average/(256/numOfBins[0]);
        atomic_fetch_add_explicit(&bins[interval], variance, memory_order_relaxed, memory_scope_work_group);
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

__kernel void CalculateAverageAllChannels(__global const int *pixels, __global double *average_y, __global double *average_u, __global double *average_v) {
    __local int blocksum_y[64];
    __local int blocksum_u[64];
    __local int blocksum_v[64];

    // Get local ids
    int l_linear_y = get_local_linear_id();

    // Get global ids
    int g_linear_y = get_global_linear_id();

    // Copy from global memory to local memory
    blocksum_y[l_linear_y] = pixels[g_linear_y];

    // Get block position and size
    int block_size = get_local_size(0) * get_local_size(1);
    int sub_block_size = block_size/4;

    // Loop summing by splitting work group in half
    for (uint stride = block_size/2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (l_linear_y < stride) {
            blocksum_y[l_linear_y] += blocksum_y[l_linear_y + stride];
        }
    }

    // Update average array
    if (l_linear_y == 0) {
        // Calculate block linear position
        average_y[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = (double)blocksum_y[0]/block_size;
    }

    // Get U and V ids
    int l_col = get_local_id(0);
    int l_row = get_local_id(1);
    
    if (l_col % 2 == 0 && l_row % 2 == 0) {
        int l_linear_u = (l_row * get_local_size(0)+ 2*l_col)/4;     //floor(l_row/2) * block_width/2 + floor(l_col/2)
        int l_linear_v = l_linear_u;

        int g_linear_u = (get_local_id(1) * get_global_size(0) + 2*get_local_id(0))/4 + (get_global_size(0) * get_global_size(1));
        int g_linear_v = g_linear_u + (get_global_size(0) * get_global_size(1))/4;

        blocksum_u[l_linear_u] = pixels[g_linear_u];
        blocksum_v[l_linear_v] = pixels[g_linear_v];
   
        for (uint stride = sub_block_size/2; stride > 0; stride /= 2) {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (l_linear_u < stride) {
                blocksum_u[l_linear_u] += blocksum_u[l_linear_u + stride];
                blocksum_v[l_linear_v] += blocksum_v[l_linear_v + stride];
            }
        }

        // Update average array
        if (l_linear_u == 0) {
            // Calculate block linear position
            average_u[get_group_id(1) * get_num_groups(0) + 2*get_group_id(0)/4] = (double)blocksum_u[0]/block_size;
        }
    }
}