#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DTYPE uint64_t
/*
 * @brief creates a "test file" by filling a bufferwith values
 */
void create_test_file(DTYPE *input, uint64_t nr_elements, DTYPE *queries, uint64_t n_queries) {

    uint64_t max = UINT64_MAX;
    uint64_t min = 0;

    srand(time(NULL));

    input[0] = 1;
    for (uint64_t i = 1; i < nr_elements; i++) {
        input[i] = input[i - 1] + (rand() % 10) + 1;
    }

    for (uint64_t i = 0; i < n_queries; i++) {
        queries[i] = input[rand() % (nr_elements - 2)];
    }
}

uint64_t binarySearch(DTYPE *input, uint64_t input_size, DTYPE *queries, unsigned n_queries) {

    uint64_t found = -1;
    uint64_t q, r, l, m;

    for (q = 0; q < n_queries; q++) {
        l = 0;
        r = input_size;
        while (l <= r) {
            m = l + (r - l) / 2;

            // Check if x is present at mid
            if (input[m] == queries[q]) {
                found += m;
                break;
            }
            // If x greater, ignore left half
            if (input[m] < queries[q])
                l = m + 1;

            // If x is smaller, ignore right half
            else
                r = m - 1;
        }
    }

    return found;
}

int main() {
    uint64_t input_size = 2048576;
    uint64_t n_queries = 16777216;

    printf("Vector size: %lu, num searches: %lu\n", input_size, n_queries);

    volatile DTYPE *input = malloc((input_size) * sizeof(DTYPE));
    volatile DTYPE *queries = malloc((n_queries) * sizeof(DTYPE));

    volatile DTYPE result_host = -1;

    // Create an input file with arbitrary data.
    create_test_file(input, input_size, queries, n_queries);

    // start_region();
    result_host = binarySearch(input, input_size - 1, queries, n_queries);
    // end_region();

    return 0;
}
