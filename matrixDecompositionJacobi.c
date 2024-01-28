/*
  Jacobi Iteration with matrix decomposition in MPI
  Name: Mursalin Sayeed
  Id: 1942882

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define tolerance 0.001
#define iterationLimit 100000

/*  x co-ordinate of this point */
double getXCoordinate(int row, int i, int localN, int processesGridLength)
{
  int n;
  double h, x;

  n = processesGridLength * (localN - 2);
  h = 1/(double)(n+1);
  return x = h * (row*(localN - 2) + i);
}

/*  y co-ordinate of this point */
double getYCoordinate(int col, int j, int localN, int processesGridLength)
{
  int n;
  double h, y;

  n = processesGridLength * (localN - 2);
  h = 1/(double)(n+1);
  return y = h * (col * (localN - 2) + j);
}

int  processRank(int row, int col, int processesGridLength)
{
  return processesGridLength * row + col;
}


double g(double x, double y)
{
  return (1+x) * sin(x+y);
}

double f(int row, int col, int i, int j, int localN, int processesGridLength)
{ 
  double x, y;

  x = getXCoordinate(row, i, localN, processesGridLength);
  y = getYCoordinate(col, j, localN, processesGridLength);
  return 2*( ((1+x)*sin(x+y)) - cos(x+y)) ;
}

void freeMemMat(double **x)
{
  free(x[0]);
  free(x);
}

/* creating matrix -- allocate m x n array as a pointer to pointer to double */
double **createMatrix(int m, int n)
{
  double **localMemLoc;
  int i;

  localMemLoc = (double **)malloc(m*sizeof(double *));
  localMemLoc[0] = (double *)malloc(m*n*sizeof(double));
  for ( i = 0; i < m; i++ )
    localMemLoc[i] = &localMemLoc[0][i*n];
  return localMemLoc;
}

// Initialize the local A matrix, b vector, and x vector
void initialize_localA(double** localA, int localN/*, double* localB, double* localX, int localN, int row, int col, int processesGridLength*/) {
  int i, j;
  // Fill localA with the stencil and localB with the function f
  for (i = 0; i < localN; i++) {
    for (j = 0; j < localN; j++) {
      if (i == j) {
        localA[i][j] = 1.0; // For the central point -> 4 / 4 = 1
      } else if (abs(i - j) == 1) {
        localA[i][j] = -0.25; // For the neighboring points -> -1 / 4 = -0.25
      } else {
        localA[i][j] = 0.0; // For all other points
      }
      //initialize localB
      // localB[i] = 0.25 * f(row, col, i, j, localN, processesGridLength); // 1 / 4 = 0.25
    }
    // Initialize localX with zeros
    // localX[i] = 0.0;
  }
}

void decomposeMatrix(double **localA, double ***localD, double ***localL, double ***localU, int localN) {
  // Allocate memory for localD, localL, and localU
  // *localD = createMatrix(localN, localN);
  // *localL = createMatrix(localN, localN);
  // *localU = createMatrix(localN, localN);

  for(int i = 0; i < localN; i++) {
    for(int j = 0; j < localN; j++) {
      if(i < j) {
        (*localU)[i][j] = localA[i][j];
      } else if(i > j) {
        (*localL)[i][j] = localA[i][j];
      } else {
        (*localD)[i][j] = 1.0;
      }
    }
  }
}

void getNewA(/*double **newA,*/ double **localA, int localN, int row, int col, int processesGridLength) {
  int i, j, n;
  double h;
  double **newA;
  newA = createMatrix(localN, localN);
  n = processesGridLength * (localN - 2);
  h = 1 / (double) (n+1);

  initialize_localA(localA, localN);

  for(i = 1; i < localN-1; i++) {
    for(j = 1; j < localN-1; j++) {
      newA[i][j] = (localA[i-1][j] + localA[i+1][j] + localA[i][j-1] + localA[i][j+1]) * 0.25;
    }
  }
  for(i = 1; i < localN-1; i++) {
    for(j = 1; j < localN-1; j++) {
      localA[i][j] = newA[i][j];
    }
  }

  freeMemMat(newA);
}

int isBoundaryPoint(int row, int col, int i, int j, int localN, int processesGridLength)
{
  if ( (row == 0 && i == 0) || (row == processesGridLength-1 && i == localN-1) )
    return 1;
  if ( (col == 0 && j == 0) || (col == processesGridLength-1 && j == localN-1) )
    return 1;
  return 0;
}

void gridInitialization(double **localA, int n, int localN, int row, int col, int processesGridLength)
{
  int i, j;
  double x, y;

  for(i = 0; i < localN; i++)
  {
    for ( j = 0; j < localN; j++ )
    {
      if ( isBoundaryPoint(row, col, i, j, localN, processesGridLength) )
      {
        y = getYCoordinate(col, j, localN, processesGridLength);
        x = getXCoordinate(row, i, localN, processesGridLength);
        localA[i][j] = g(x, y);
      }
      else
        localA[i][j] = 0.0;
    }
  }
}

void printVector(double *vector, int size) {
  for (int i = 0; i < size; i++) {
    printf("%f ", vector[i]);
  }
  printf("\n");
}

void printMatrix(double **matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%f ", matrix[i][j]);
    }
    printf("\n");
  }
}

void getB(double *localB, int row, int col, int localN, int processesGridLength) {
  for (int i = 0; i < localN; i++) {
    localB[i] = 0.25 * f(row, col, i, i, localN, processesGridLength); // 1 / 4 = 0.25
  }
}

void jacobiIteration(int localN, double **localA, int processesGridLength, int row, int col, /*double **newA,*/ double **localD, double **localL, double **localU, double *localB, double *localX, int myRank) {
  // double *newX = malloc(localN * sizeof(double));
  double sum;
  double norm;
  double globalNorm;
  int iteration = 0;

  // exchangeData(row, col, localN, newA, localA, processesGridLength, tag1, tag2, &status);
  getNewA(localA, localN, row, col, processesGridLength);
  // printf("LOCAL A FROM RANK %d", myRank);
  // printMatrix(localA, localN, localN);
  decomposeMatrix(localA, &localD, &localL, &localU, localN);
  // printf("LocalD from process %d\n", myRank);
  // printMatrix(localD, localN, localN);
  // printf("LocalL from process %d\n", myRank);
  // printMatrix(localL, localN, localN);
  // printf("LocalU from process %d\n", myRank);
  // printMatrix(localU, localN, localN);

  getB(localB, row, col, localN, processesGridLength);
  // printf("LocalB from process %d\n", myRank);
  // printVector(localB, localN);

  // Initialize localX with random values
  for (int i = 0; i < localN; i++) {
    localX[i] = 0.0;
  }
  // printf("***BEFORE LocalX from process %d\n", myRank);
  // printVector(localX, localN);

  do {
    iteration++;
    // Copy newX to localX
    // memcpy(localX, newX, localN * sizeof(double));

    for (int i = 0; i < localN; i++) {
      sum = 0.0;

      // Calculate the sum of U and L times the old x
      for (int j = 0; j < localN; j++) {
        if (j != i) {
          sum += fabs(localL[i][j] + localU[i][j]) * localX[j];
          // printf("Rank: %d, Iteration: %d, Sum: %f\n", myRank, iteration, sum);
        }
      }

      // Update newX
      localX[i] = fabs(localB[i] - sum); // localD[i][i]
      // printf("Rank: %d, Iteration: %d, LocalX[%d]: %f\n", myRank, iteration, i, localX[i]);
    }

    // Calculate the norm of the residual
    double *residual = malloc(localN * sizeof(double));
    norm = 0.0;

    // Calculate the residual vector
    for (int i = 0; i < localN; i++) {
      double Ax = 0.0;
      for (int j = 1; j < localN; j++) {
        Ax += localA[i][j] * localX[j];
      }
      residual[i] = fabs(localB[i] - Ax);
      // printf("Rank: %d, Iteration: %d, Residual[%d]: %f\n", myRank, iteration, i, residual[i]);
    }

    // Calculate the norm of the residual vector
    for (int i = 0; i < localN; i++) {
      norm += fabs(residual[i] * residual[i]);
      // printf("Rank: %d, Iteration: %d, Norm: %f\n", myRank, iteration, norm);
    }
    
    if (norm <= 0.0) {
      norm = 0.0;
    } else {
      norm = sqrt(norm);
    }
    
    MPI_Allreduce(&norm, &globalNorm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // printf("Rank: %d, Iteration: %d, Norm: %f, Global_Norm: %f\n", myRank, iteration, norm, globalNorm);
    // printf("LocalX from process %d\n", myRank);
    // printVector(localX, localN);
    // printf("LocalB from process %d\n", myRank);
    // printVector(localB, localN);
    // printf("LocalA from process %d\n", myRank);
    // printMatrix(localA, localN, localN);

    free(residual);
  } while ( iteration < iterationLimit && tolerance < globalNorm);

  if(processRank(row,col, processesGridLength) == 0 && tolerance > globalNorm )
    printf("elapsed iteration: %d, norm: %f\n", iteration, globalNorm);
  else
    printf("Jacobi iteration did not converge. Iteration: %d, Norm: %f\n", iteration, globalNorm);

  // free(newX);

}


int main(int argc, char **argv)
{
  /* processesGridLength = number of block rows = number of block columns 
     p = total no. porcesses in grid
  */
  
  int row;
  int col;
  int myRank;
  int processesGridLength;
  int globalN = 400;
  int p;
  int dims[2];
  int wrap_around[2];
  int coords[2];
  int localN;

  double startTime, elapsedTime;

  MPI_Comm comm_2D;
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &p);
  processesGridLength = sqrt(p);

  localN = globalN / processesGridLength;

  /* Set up cartesian m x m grid */
  dims[0] = dims[1] = processesGridLength;        
  wrap_around[0] = wrap_around[1] = 0; 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, wrap_around, 1, &comm_2D);
  MPI_Comm_rank(comm_2D, &myRank);
  MPI_Cart_coords(comm_2D, myRank, 2, coords);
  row = coords[0];
  col = coords[1];

  printf("Processes: %d\n", p);
  printf("ProcessesGridLength: %d\n", processesGridLength);
  printf("LocalN: %d\n", localN);
  printf("Rank: %d, Row: %d, Col: %d\n", myRank, row, col);

  double** localA;
  localA = createMatrix(localN, localN);

  // gridInitialization(localA, globalN, localN, row, col, processesGridLength);

  double* localB;
  localB = (double*)malloc(localN * sizeof(double));
  double* localX;
  localX = (double*)malloc(localN * sizeof(double));
  
  // initialize_local_A_b_x(localA, localB, localX, localN, row, col, processesGridLength);

  MPI_Barrier(MPI_COMM_WORLD);

  // printf("LocalA from process %d\n", myRank);
  // printMatrix(localA, localN, localN);
  // printf("LocalB from process %d\n", myRank);
  // printVector(localB, localN);
  // printf("LocalX from process %d\n", myRank);
  // printVector(localX, localN);

  // Decompose the local A matrix into localD, localL, and localU
  double **localD;
  double **localL;
  double **localU;
  localD = createMatrix(localN, localN);
  localL = createMatrix(localN, localN);
  localU = createMatrix(localN, localN);

  MPI_Barrier(MPI_COMM_WORLD);

  // decomposeMatrix(localA, &localD, &localL, &localU, localN);

  // printf("LocalD from process %d\n", myRank);
  // printMatrix(localD, localN, localN);
  // printf("LocalL from process %d\n", myRank);
  // printMatrix(localL, localN, localN);
  // printf("LocalU from process %d\n", myRank);
  // printMatrix(localU, localN, localN);

  // double **newA;
  // newA = createMatrix(localN, localN);

  startTime = MPI_Wtime();

  jacobiIteration(localN, localA, processesGridLength, row, col, /*newA,*/ localD, localL, localU, localB, localX, myRank);

  elapsedTime = MPI_Wtime() - startTime;

  MPI_Barrier(MPI_COMM_WORLD);

  /* Clear allocated memory */
  freeMemMat(localA);
  freeMemMat(localD);
  freeMemMat(localL);
  freeMemMat(localU);
  free(localB);
  free(localX);                   

  if(myRank == 0)
    printf("Elapsed time: %e\n", elapsedTime); 
    
  /* Clear communicator */ 
  MPI_Comm_free(&comm_2D);   
  MPI_Finalize();
} 