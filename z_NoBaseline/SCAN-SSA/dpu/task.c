/*
 * Scan with multiple tasklets (Scan-scan-add)
 *
 */
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <handshake.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Array for communication between adjacent tasklets
T message[NR_TASKLETS];
T message_partial_count;

// Scan in each tasklet
static T scan(T *output, T *input) {
    output[0] = input[0];
#pragma unroll
    for (unsigned int j = 1; j < REGS; j++) {
        output[j] = output[j - 1] + input[j];
    }
    return output[REGS - 1];
}

// Handshake with adjacent tasklets
static T handshake_sync(T l_count, unsigned int tasklet_id) {
    T p_count;
    // Wait and read message
    if (tasklet_id != 0) {
        handshake_wait_for(tasklet_id - 1);
        p_count = message[tasklet_id];
    } else
        p_count = 0;
    // Write message and notify
    if (tasklet_id < NR_TASKLETS - 1) {
        message[tasklet_id + 1] = p_count + l_count;
        handshake_notify();
    }
    return p_count;
}

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// Add in each tasklet
static void add(T *output, T p_count) {
#pragma unroll
    for (unsigned int j = 0; j < REGS; j++) {
        output[j] += p_count;
    }
}

extern int main_kernel1(void);
extern int main_kernel2(void);

int (*kernels[nr_kernels])(void) = {main_kernel1, main_kernel2};

int main(void) {
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

// Scan-(handshake)scan
int main_kernel1() {
#if 1 // Comment out for appendix experiment
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0) { // Initialize once the cycle counter
        mem_reset();       // Reset the heap
    }
    // Barrier
    barrier_wait(&my_barrier);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes);

    // Initialize a local cache to store the MRAM block
    T *cache_A = (T *)mem_alloc(BLOCK_SIZE);
    T *cache_B = (T *)mem_alloc(BLOCK_SIZE);

    // Initialize shared variable
    if (tasklet_id == NR_TASKLETS - 1)
        message_partial_count = DPU_INPUT_ARGUMENTS.t_count;
    // Barrier
    barrier_wait(&my_barrier);

    for (unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS) {

        // Load cache with current MRAM block
        mram_read((const __mram_ptr void *)(mram_base_addr_A + byte_index), cache_A, BLOCK_SIZE);

        // Scan in each tasklet
        T l_count = scan(cache_B, cache_A);

        // Sync with adjacent tasklets
        T p_count = handshake_sync(l_count, tasklet_id);

        // Barrier
        barrier_wait(&my_barrier);

        // Add in each tasklet
        add(cache_B, message_partial_count + p_count);

        // Write cache to current MRAM block
        mram_write(cache_B, (__mram_ptr void *)(mram_base_addr_B + byte_index), BLOCK_SIZE);

        // Total count in this DPU
        if (tasklet_id == NR_TASKLETS - 1) {
            result->t_count = message_partial_count + p_count + l_count;
            message_partial_count = result->t_count;
        }
    }

#endif
    return 0;
}

// Add
int main_kernel2() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0) { // Initialize once the cycle counter
        mem_reset();       // Reset the heap
    }
    // Barrier
    barrier_wait(&my_barrier);

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes);

    // Initialize a local cache to store the MRAM block
    T *cache_A = (T *)mem_alloc(BLOCK_SIZE);

    T t_count = DPU_INPUT_ARGUMENTS.t_count;

    for (unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS) {

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const *)(mram_base_addr_B + byte_index), cache_A, BLOCK_SIZE);

        // Add in each tasklet
        add(cache_A, t_count);

        // Write cache to current MRAM block
        mram_write(cache_A, (__mram_ptr void *)(mram_base_addr_B + byte_index), BLOCK_SIZE);
    }

    return 0;
}
