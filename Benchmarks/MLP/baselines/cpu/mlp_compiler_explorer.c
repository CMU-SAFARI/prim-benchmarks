#define T volatile int32_t
#define NUM_LAYERS 3
#define max(x, y) (x > y ? x : y)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

T **A;
T *B;
T *C;

// Create input arrays
static void init_data(T **A, T *B, unsigned int m_size, unsigned int n_size) {
    for (unsigned int l = 0; l < NUM_LAYERS; l++)
        for (unsigned int i = 0; i < m_size * n_size; i++) {
            if (i % 100 < 98) {
                A[l][i] = 0;
            } else {
                A[l][i] = (l + i) % 2;
            }
        }
    for (unsigned int i = 0; i < n_size; i++) {
        if (i % 50 < 48) {
            B[i] = 0;
        } else {
            B[i] = i % 2;
        }
    }
}

// Compute output in the host
static void mlp_host(T *C, T **A, T *B, unsigned int m_size, unsigned int n_size) {
    for (unsigned int nl = 0; nl < NUM_LAYERS; nl++) {
        for (unsigned int m = 0; m < m_size; m++) {
            C[m] = 0;
        }
        for (unsigned int m = 0; m < m_size; m++) {
            for (unsigned int n = 0; n < n_size; n++) {
                C[m] += A[nl][m * n_size + n] * B[n];
            }
            C[m] = max(0, C[m]);
        }
        for (unsigned int n = 0; n < n_size; n++) {
            B[n] = C[n];
        }
    }
}

static uint64_t mlp_host_sum(uint64_t n_size, uint64_t m_size) {
    uint64_t sum = 0;
    for (uint64_t m = 0; m < n_size; m++) {
        sum += B[m];
    }
    return sum;
}

int main() {
    uint64_t n_size = 8192;
    uint64_t m_size = 20480;

    A = (T **)malloc(NUM_LAYERS * sizeof(T *));
    for (int l = 0; l < NUM_LAYERS; l++)
        A[l] = (T *)malloc(n_size * m_size * sizeof(unsigned int));
    B = (T *)malloc(m_size * sizeof(unsigned int));
    C = (T *)malloc(m_size * sizeof(unsigned int));

    // Create an input file with arbitrary data.
    init_data(A, B, m_size, n_size);

    mlp_host(C, A, B, n_size, m_size);

    uint32_t sum = mlp_host_sum(n_size, m_size);

    printf("SUM = %d \n", sum);

    // for (int l = 0; l < NUM_LAYERS; l++)
    //     free(A[l]);
    // free(A);
    // free(B);
    // free(C);

    return 0;
}
