/*
 * Arithmetic Throughtput versus Operational Intensity
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

// Update
static void update(T *bufferA, T scalar, uint32_t rep, uint32_t str) {
    for (unsigned int r = 0; r < rep; r++) {
        for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i += str) {
#ifdef ADD
            bufferA[i] += scalar; // ADD
#elif SUB
            bufferA[i] -= scalar; // SUB
#elif MUL
            bufferA[i] *= scalar; // MUL
#elif DIV
            bufferA[i] /= scalar; // DIV
#endif
        }
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
    // Barrier
    barrier_wait(&my_barrier);
    perfcounter_cycles cycles;

    uint32_t input_size_dpu = DPU_INPUT_ARGUMENTS.size / sizeof(T);

    uint32_t rep = DPU_INPUT_ARGUMENTS.repetitions;
    uint32_t str = DPU_INPUT_ARGUMENTS.stride;
    T scalar = DPU_INPUT_ARGUMENTS.scalar; // Simply use this number as a scalar

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->cycles = 0;

    // Address of the current processing block in MRAM
    uint32_t mram_base_addr_A = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id << BLOCK_SIZE_LOG2));
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id << BLOCK_SIZE_LOG2) + input_size_dpu * sizeof(T));

    // Initialize a local cache to store the MRAM block
    T *cache_A = (T *)mem_alloc(BLOCK_SIZE);

    barrier_wait(&my_barrier);
    timer_start(&cycles); // START TIMER

    for (unsigned int byte_index = 0; byte_index < input_size_dpu * sizeof(T); byte_index += BLOCK_SIZE * NR_TASKLETS) {

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const *)(mram_base_addr_A + byte_index), cache_A, BLOCK_SIZE);

        // Update
        update(cache_A, scalar, rep, str);

        // Write cache to current MRAM block
        mram_write(cache_A, (__mram_ptr void *)(mram_base_addr_B + byte_index), BLOCK_SIZE);
    }

    result->cycles = timer_stop(&cycles); // STOP TIMER
    barrier_wait(&my_barrier);

    return 0;
}
