#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define VECTOR_SIZE 5000
#define NUM_ITERATIONS 1000000

// Dot product function written in C
float dot_product_c(const float* a, const float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Dot product function using vdpps with inline assembly
float dot_product_dpps(const float* a, const float* b, int size) {
    float result = 0.0f;
    int i;
    for (i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vdp = _mm256_dp_ps(va, vb, 0xFF);
        result += _mm256_cvtss_f32(vdp);
        result += _mm_cvtss_f32(_mm256_extractf128_ps(vdp, 1));
    }
    // Handle remaining elements
//    for (; i < size; i++) {
//        result += a[i] * b[i];
//    }
    return result;
}

int main() {
    float* a = (float*)aligned_alloc(16, VECTOR_SIZE * sizeof(float));
    float* b = (float*)aligned_alloc(16, VECTOR_SIZE * sizeof(float));

    // Initialize random seed
    srand(time(NULL));

    // Generate random vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // Print the generated vectors
    printf("Vector a: [");
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%.6f", a[i]);
        if (i < VECTOR_SIZE - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    printf("Vector b: [");
    for (int i = 0; i < VECTOR_SIZE; i++) {
        printf("%.6f", b[i]);
        if (i < VECTOR_SIZE - 1) {
            printf(", ");
        }
    }
    printf("]\n");

    printf("Processing...\n");

    // Benchmark dot product function written in C
    clock_t start = clock();
    float result_c;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result_c = dot_product_c(a, b, VECTOR_SIZE);
    }
    clock_t end = clock();
    double time_c = (double)(end - start) / CLOCKS_PER_SEC;

    // Benchmark dot product function using AVX intrinsics
    start = clock();
    float result_avx;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        result_avx = dot_product_dpps(a, b, VECTOR_SIZE);
    }
    end = clock();
    double time_sse = (double)(end - start) / CLOCKS_PER_SEC;

    // Print the calculated dot products
    printf("Dot product (C): %.9f\n", result_c);
    printf("Dot product (AVX): %.9f\n", result_avx);

    // Print the execution times
    printf("Execution time (C): %.6f seconds\n", time_c);
    printf("Execution time (AVX): %.6f seconds\n", time_sse);

    free(a);
    free(b);

    return 0;
}