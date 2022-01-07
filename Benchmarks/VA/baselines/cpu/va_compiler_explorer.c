#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

volatile static int32_t *A;
volatile static int32_t *B;
volatile static int32_t *C;

int main() {
    unsigned int nr_elements = 1024;

    for (int i = 0; i < nr_elements; i++) {
        C[i] = A[i] + B[i];
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
