#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define VECTOR_SIZE 3
#define NUM_ITERATIONS 100000

// Dot product function written in C
float dot_product_c(const float* a, const float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Dot product function using AVX intrinsics
float dot_product_avx(const float* a, const float* b, int size) {
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 vmul = _mm_mul_ps(va, vb);
    __m128 vsum = _mm_hadd_ps(vmul, vmul);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
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
    printf("Vector a: [%.6f, %.6f, %.6f]\n", a[0], a[1], a[2]);
    printf("Vector b: [%.6f, %.6f, %.6f]\n", b[0], b[1], b[2]);

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
        result_avx = dot_product_avx(a, b, VECTOR_SIZE);
    }
    end = clock();
    double time_avx = (double)(end - start) / CLOCKS_PER_SEC;

    // Print the calculated dot products
    printf("Dot product (C): %.6f\n", result_c);
    printf("Dot product (AVX): %.6f\n", result_avx);

    // Print the execution times
    printf("Execution time (C): %.6f seconds\n", time_c);
    printf("Execution time (AVX): %.6f seconds\n", time_avx);

    free(a);
    free(b);

    return 0;
}