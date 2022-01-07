#include <stdio.h>
#include <stdlib.h>

#include "../../support/timer.h"
#include "gemv_utils.h"
#include "shared.hpp"

int main(int argc, char *argv[]) {
    const size_t rows = 128;
    const size_t cols = 64;

    double **A, *b, *x;

    b = (double *)malloc(sizeof(double) * rows);
    x = (double *)malloc(sizeof(double) * cols);

    allocate_dense(rows, cols, &A);

    make_hilbert_mat(rows, cols, &A);

    {
        for (size_t i = 0; i < cols; i++) {
            x[i] = (double)i + 1;
        }

        for (size_t i = 0; i < rows; i++) {
            b[i] = (double)0.0;
        }
    }

    Timer timer;
    start(&timer, 0, 0);

    start_region();
    gemv(A, x, rows, cols, &b);
    end_region();

    stop(&timer, 0);

    printf("Kernel ");
    print(&timer, 0, 1);
    printf("\n");

#if 0
  print_vec(x, rows);
  print_mat(A, rows, cols);
  print_vec(b, rows);
#endif

    printf("sum(x) = %f, sum(Ax) = %f\n", sum_vec(x, cols), sum_vec(b, rows));
    return 0;
}

void gemv(double **A, double *x, size_t rows, size_t cols, double **b) {
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++) {
            (*b)[i] = (*b)[i] + A[i][j] * x[j];
        }
}

void make_hilbert_mat(size_t rows, size_t cols, double ***A) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            (*A)[i][j] = 1.0 / ((double)i + (double)j + 1.0);
        }
    }
}

double sum_vec(double *vec, size_t rows) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++)
        sum = sum + vec[i];
    return sum;
}
