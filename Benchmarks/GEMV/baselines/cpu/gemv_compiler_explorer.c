#include <stdio.h>
#include <stdlib.h>

void allocate_dense(size_t rows, size_t cols, double ***dense) {

    *dense = malloc(sizeof(double) * rows);
    **dense = malloc(sizeof(double) * rows * cols);

    for (size_t i = 0; i < rows; i++) {
        (*dense)[i] = (*dense)[0] + i * cols;
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

int main() {
    const size_t rows = 128;
    const size_t cols = 64;

    double **A, *x;
    volatile double *b;

    b = (double *)malloc(sizeof(double) * rows);
    x = (double *)malloc(sizeof(double) * cols);

    allocate_dense(rows, cols, (double ***)&A);

    make_hilbert_mat(rows, cols, (double ***)&A);

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++) {
            b[i] = b[i] + A[i][j] * x[j];
        }
}