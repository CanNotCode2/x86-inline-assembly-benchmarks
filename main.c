#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define VECTOR_SIZE 1024
#define NUM_ITERATIONS 100000

// Dot product function written in C
float dot_product_c(const float* a, const float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Dot product function using vdpps with inline assembly
float dot_product_vdpps(const float* a, const float* b, int size) {
    float result = 0.0f;
    int i;
    for (i = 0; i < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vdp = _mm256_dp_ps(va, vb, 0xFF);
        result += _mm_cvtss_f32(_mm256_castps256_ps128(vdp));
    }
    // Handle remaining elements
    for (; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    float* a = (float*)aligned_alloc(32, VECTOR_SIZE * sizeof(float));
    float* b = (float*)aligned_alloc(32, VECTOR_SIZE * sizeof(float));

    // Initialize random seed
    srand(time(NULL));

    // Generate random vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // Benchmark dot product function written in C
    clock_t start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float result = dot_product_c(a, b, VECTOR_SIZE);
    }
    clock_t end = clock();
    double time_c = (double)(end - start) / CLOCKS_PER_SEC;

    // Benchmark dot product function using vdpps with inline assembly
    start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float result = dot_product_vdpps(a, b, VECTOR_SIZE);
    }
    end = clock();
    double time_vdpps = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Dot product (C): %.6f seconds\n", time_c);
    printf("Dot product (vdpps): %.6f seconds\n", time_vdpps);

    free(a);
    free(b);

    return 0;
}