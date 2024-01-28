#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define M 20

void read_vector(double** v, char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Cannot open file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    *v = (double*)malloc(sizeof(double) * M);
    for (int i = 0; i < M; i++) {
        fscanf(file, "%lf", &(*v)[i]);  // Read each element of the vector
    }

    fclose(file);
}

void distribute_vector(double** v, int rank, int size) {
    int local_n = M / size;  // Calculate the number of elements for each process

    if (rank != 0) {
        *v = (double*)malloc(sizeof(double) * local_n);  // Allocate memory for the vector in other processes
    }

    MPI_Scatter(*v, local_n, MPI_DOUBLE, *v, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // Scatter the vector
}

double dot_product(double* v, double* w, int n) {
    // printf("Process %d: ", n);
    // printf("\n");
    // printf("Vector v: ");
    // for (int i = 0; i < M; i++) {
    //     printf("%f ", v[i]);
    // }
    // printf("\n");

    // printf("Vector w: ");
    // for (int i = 0; i < M; i++) {
    //     printf("%f ", w[i]);
    // }
    // printf("\n");
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * w[i];
    }
    double result;
    MPI_Allreduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // printf("Sum: %f\n", sum);
    // printf("Result:  %f\n", result);
    return result;
}

double vector_norm(double* v, int n) {
    return sqrt(dot_product(v, v, n));
}

void scalar_multiply(double* v, double a, int n) {
    for (int i = 0; i < n; i++) {
        v[i] *= a;
    }
}

void vector_add_scalar_multiply(double* v, double* w, double a, int n) {
    for (int i = 0; i < n; i++) {
        v[i] += a * w[i];
    }
}

void gramm_schmidt(double** x, double** q, double** r, int n, int m) {
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < j; i++) {
            r[i][j] = dot_product(x[j], q[i], n);
            vector_add_scalar_multiply(x[j], q[i], -r[i][j], n);
        }
        r[j][j] = vector_norm(x[j], n);
        if (r[j][j] == 0) {
            break;
        }
        scalar_multiply(x[j], 1 / r[j][j], n);
        q[j] = (double*)malloc(sizeof(double) * n);  // Allocate new memory for q[j]
        for (int i = 0; i < n; i++) {
            q[j][i] = x[j][i];  // Copy the elements of x[j] to q[j]
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n_bar = M / size;
    double* v1;
    // double* v2;

    if (rank == 0) {
        read_vector(&v1, "vector1.dat");  // Only process 0 reads the first vector
        // read_vector(&v2, "vector2.dat");  // Only process 0 reads the second vector
    }

    distribute_vector(&v1, rank, size);  // Distribute the first vector to all processes
    // distribute_vector(&v2, rank, size);  // Distribute the second vector to all processes

    // double result = dot_product(v1, v2, n_bar);  // Compute the dot product of the two vectors

    // if (rank == 0) {
    //     printf("Dot product: %f\n", result);  // Only process 0 prints the result
    // }

    // Allocate memory for the vectors q and r
    double** q = malloc(M * sizeof(double*));
    double** r = malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        q[i] = malloc(M * sizeof(double));
        r[i] = malloc(M * sizeof(double));
    }

    // Perform Gramm-Schmidt orthogonalization
    gramm_schmidt(&v1, q, r, M, 1);

    // Deallocate memory
    for (int i = 0; i < M; i++) {
        free(q[i]);
        free(r[i]);
    }
    free(q);
    free(r);

    free(v1);
    MPI_Finalize();

    return 0;
}