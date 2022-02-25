/*
 * STREAM Copy (WRAM)
 *
 */
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "../support/common.h"
#include "../support/cyclecount.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Copy
static void copyw_dpu(T *bufferB, T *bufferA) {

#pragma unroll
    for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
        bufferB[i] = bufferA[i];
    }
}

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) {
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0) { // Initialize once the cycle counter
        mem_reset();       // Reset the heap

        perfcounter_config(COUNT_CYCLES, true);
    }
    perfcounter_cycles cycles;
    // Barrier
    barrier_wait(&my_barrier);
#ifndef WRAM
    timer_start(&cycles); // START TIMER
#endif

    uint32_t input_size_dpu = DPU_INPUT_ARGUMENTS.size / sizeof(T);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->cycles = 0;

    // Address of the current processing block in MRAM
    uint32_t mram_base_addr_A = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id << BLOCK_SIZE_LOG2));
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id << BLOCK_SIZE_LOG2) + input_size_dpu * sizeof(T));

    // Initialize a local cache to store the MRAM block
    T *cache_A = (T *)mem_alloc(BLOCK_SIZE);
    T *cache_B = (T *)mem_alloc(BLOCK_SIZE);

    for (unsigned int byte_index = 0; byte_index < input_size_dpu * sizeof(T); byte_index += BLOCK_SIZE * NR_TASKLETS) {

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const *)(mram_base_addr_A + byte_index), cache_A, BLOCK_SIZE);

#ifdef WRAM
        // Barrier
        barrier_wait(&my_barrier);
        timer_start(&cycles); // START TIMER
#endif

        // Copy
        copyw_dpu(cache_B, cache_A);

#ifdef WRAM
        result->cycles += timer_stop(&cycles); // STOP TIMER
        // Barrier
        barrier_wait(&my_barrier);
#endif

        // Write cache to current MRAM block
        mram_write(cache_B, (__mram_ptr void *)(mram_base_addr_B + byte_index), BLOCK_SIZE);
    }

#ifndef WRAM
    result->cycles = timer_stop(&cycles); // STOP TIMER
#endif
    return 0;
}
