#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#include "MatUtil.h"

#define MPI_TYPE MPI_INT
#define MIN(a, b) (a != -1 && a <= b) ? a : b

void PT_APSP(int*, int, int, int);

int main(int argc, char **argv)
{
	int rank;      /* Process rank */
	int numProc;         /* Number of processes */
	int rows;		/* Number of rows per stripe */
		
	double time, max_time;

	size_t N;		/* Size of data/number of vertices */
	int* mat;		/* Original data matrix */
	int* ref;		/* Reference matrix using serial algorithm*/
	int* result;	/* Result matrix using parallel algorithm*/
	int* pStripe;	/* Memory of each stripe */
	
	struct timeval tv1, tv2;

	// initial MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);
	
	if (rank == 0){
		if(argc != 2){
			printf("Usage: test {N}\n");
			exit(-1);
		}
		N = atoi(argv[1]);
		printf("size = %d\n", N);		
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	rows = N / numProc;
	
	if (rank == 0){
		// generate a random matrix.
		mat = (int*)malloc(sizeof(int)*N*N);
		GenMatrix(mat, N);

		// compute the reference result.
		ref = (int*)malloc(sizeof(int)*N*N);
		memcpy(ref, mat, sizeof(int)*N*N);
		gettimeofday(&tv1, NULL);
		ST_APSP(ref, N);
		gettimeofday(&tv2, NULL);
		printf("Serial, matrix size %d : %d usecs\n", N, 
			        ((tv2.tv_sec-tv1.tv_sec) * 1000000+tv2.tv_usec-tv1.tv_usec));

		// generate result matrix for computation
		result = (int*)malloc(sizeof(int)*N*N);
		memcpy(result, mat, sizeof(int)*N*N);
		gettimeofday(&tv1, NULL);
	}
	// start the timer
	//time = MPI_Wtime();
	
	// allocate memory for the current stripe
	pStripe = (int*)malloc(sizeof(int) * N * rows);	
	MPI_Barrier(MPI_COMM_WORLD);
	// distribute data among processes
	MPI_Scatter(result, rows * N, MPI_INT, pStripe, rows * N, MPI_INT, 0, MPI_COMM_WORLD);
	
	// compute your results
	// parallel floyd algorithm
	PT_APSP(pStripe, N, rows, rank);
	//MPI_Barrier(MPI_COMM_WORLD);
	
	// collect result
	MPI_Gather(pStripe, rows * N, MPI_INT, result, rows * N, MPI_INT, 0, MPI_COMM_WORLD);
	
	//time = MPI_Wtime() - time;
	MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (!rank) {
		gettimeofday(&tv2, NULL);
		printf ("Floyd, matrix size %d, %d processes: %d usecs\n", N, numProc, ((tv2.tv_sec-tv1.tv_sec) * 1000000+tv2.tv_usec-tv1.tv_usec));
		// compare your result with reference result
		if(CmpArray(result, ref, N*N))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
	}
	
	MPI_Finalize();
	free(mat);
	free(result);
	free(ref);
}

void PT_APSP(int* pStripe, int size, int rows, int rank)
{
	int  i, j, k;
	int  offset;   //Local index of broadcast row
	int  root;     //Process controlling row to be bcast
	int* tmp;      //Holds the broadcast row

	tmp = (int*) malloc (size * sizeof(int));
	printf ("matrix size %d, %d\n", size, rank);
	
	for (k = 0; k < size; k++) {
		root = k / rows; //evaluates to the rank of the process controlling that element of the array.
		if (root == rank) {
			offset = k - rank * rows; //expands to an expression whose value is the first, or lowest, index controlled by the process.
	        	for (j = 0; j < size; j++){
	        		tmp[j] = pStripe[offset * size + j];
			}
		}
		MPI_Bcast (tmp, size, MPI_TYPE, root, MPI_COMM_WORLD);

		for (i = 0; i < rows; i++)
	        	for (j = 0; j < size; j++)
				if ((pStripe[i * size + k] != -1) && (tmp[j] != -1))
	        			pStripe[i * size + j] = MIN(pStripe[i * size + j],pStripe[i * size + k]+tmp[j]);
	}
	free (tmp);
}
