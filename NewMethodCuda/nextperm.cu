#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
#include "EnumaratorSequence.cu"
#include <time.h>

//#define NUM_THREADS 1024
#define THREAD_PER_BLOCK 128
#define N 20
#define NUM_DEVICES 2

__device__
void printVector(char* array, int length){
	for(int k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

__device__
void inversion(char* vet, int length){
	char temp;
	for(int i = 0; i < length/2; i++){
		temp = vet[length-i-1];
		vet[length-i-1] = vet[i];
		vet[i] = temp;
	}	
}

__device__
void rotation(char *array, int length){
  char temp;
  int i;
  temp = array[0];
  for (i = 0; i < length-1; i++)
     array[i] = array[i+1];
  array[i] = temp;
}

__device__
void insert (char *array, int length, char elem, int pos) {
    int i = length-1;
    for (; i > pos; --i) {
        array[i] = array[i-1];
    }
    array[pos] = elem;
}

__device__
void shiftRight (char *array, int length) {
    char tmp = array[0];
    array[0] = array[length-1];
    int i = length-1;
    for (; i > 1; --i) {
        array[i] = array[i-1];
    }
    array[1] = tmp;
}

unsigned long long fatorialHost(unsigned long long n){
	int i;
	unsigned long long result = 1;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

//Calcula o LIS de todo o conjunto R partindo do pivor principal da ordem lexico gráfica
//Caso encontre um valor que é menor do que o máximo local de S, então ele retorna e não faz os outros calculos.
__global__
void decideLS(char* d_lMax_S, char* d_lMax_SP1, int length, unsigned long long maxSeq, int numThreads, int initThread){
	extern __shared__ char s_vet[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 	
	int s_index = (length+1)*threadIdx.x; //Indice da shared memory
	unsigned long long int indexSeq = tid+initThread;

	//Esses dois vetores são utilizados no LIS e no LDS, são declarados do lado de fora para
	//gastar menos memória e não ter necessidade de dar malloc.
	char MP[N*(N+1)/2]; //Vetor de most promising
	char last[N]; //Vetor de last de MP
	//Valores com os resultados encontrados no LIS e no LDS
	char lLIS, lLDS;
	char lMin_R;
	char lMin_RP1;
    unsigned int expectedValue = ceil(length/3.0);
	while(indexSeq < maxSeq){
		getSequence(s_vet + s_index, length, indexSeq);
        /*if (tid == 7) {
            printVector(s_vet+s_index, length);
        }*/
		lMin_R = 20; //Variavel que representa o min encontrado no conjunto R
		for(int i = 0; i < length; i++){ //Rotação
			lLIS = LIS(s_vet + s_index, last, MP, length);
			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLIS < lMin_R){
				lMin_R = lLIS;	
			}

			lLDS = LDS(s_vet + s_index, last, MP, length);
			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLDS < lMin_R){
				lMin_R = lLDS;	
			}

			rotation(s_vet + s_index, length);
		}

        /*if (tid == 7) {
        }*/
		//Caso o resultado final encontrado de R chegue ate o final, então significa que ele é maior
		//Que o minimo local encontrado até o momento.
        if (d_lMax_S[tid] < lMin_R) {
            d_lMax_S[tid] = lMin_R;
        }

        if (lMin_R >= expectedValue) {
            int pos = 1;
            insert(s_vet+s_index, length+1, length+1, pos);
            for (; pos < length+1; pos++) {
                lMin_RP1 = 20; //Variavel que representa o min encontrado no conjunto R
                for(int i = 0; i < length+1; i++){ //Rotação
                    lLIS = LIS(s_vet + s_index, last, MP, length+1);
                    //caso seja menor que o minimo do conjunto R, então modificar o valor
                    if(lLIS < lMin_RP1){
                        lMin_RP1 = lLIS;	
                    }

                    lLDS = LDS(s_vet + s_index, last, MP, length+1);
                    //caso seja menor que o minimo do conjunto R, então modificar o valor
                    if(lLDS < lMin_RP1){
                        lMin_RP1 = lLDS;	
                    }

                    rotation(s_vet + s_index, length+1);
                }
                if (d_lMax_SP1[tid] < lMin_RP1) {
                    d_lMax_SP1[tid] = lMin_RP1;
                }
                shiftRight(s_vet + s_index + 1, length-1);
            }
        }

        //printf("tid = %d, maxS= %d, index %d\n", tid, d_lMax_S[tid], indexSeq);
		indexSeq += numThreads;
	}
	__syncthreads();
	if (tid == 0) {
		char lMaxS = d_lMax_S[0];
		char lMaxSP1 = d_lMax_SP1[0];
		for(int i = 0; i < 10240; i++){
			//printf("DLMaxS %d . %d\n",i, d_lMax_S[i]);
			if(lMaxS < d_lMax_S[i]){
				lMaxS = d_lMax_S[i];
			}
			if(lMaxSP1 < d_lMax_SP1[i]){
				lMaxSP1 = d_lMax_SP1[i];
			}
		}
		//printf("LmaxS %d\n", lMaxS);
		d_lMax_S[0] = lMaxS;
		d_lMax_SP1[0] = lMaxSP1;
	}
}

//Com os valores de máximos locais de S, calcular o máximo global.
void calcLMaxGlobalS(char* lMax_globalS, char* lMax_localS, int tamVec){
	//Número de conjuntos
	for(int i = 0; i < tamVec; i++){
		//printf("%d\n", lMax_localS[i]);
		if(*lMax_globalS < lMax_localS[i]){
			*lMax_globalS = lMax_localS[i];
		}
	}
}

//Seja S o conjunto de todas las sequencias dos n primeiros números naturais.
//Defina R(s), com s \in S o conjunto de todas as sequencias que podem
//ser geradas rotacionando S.
//Defina LIS(s) e LDS(s) como você sabe e sejam |LIS(s)| e |LDS(s)| suas
//cardinalidades.
//Determinar Max_{s \in S}(Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)))
int main(int argc, char *argv[]){
	//char* h_sequence;            //Vetor com a sequência pivor do grupo
	//char* h_threadSequences;      //Vetor com as sequências criadas
	char* d_lMax_localS0;      //Vetor com os máximos locais de S, cada thread tem um máximo local
	char* d_lMax_localS1;
	char* d_lMax_localS0P1;
	char* d_lMax_localS1P1;
	char* h_lMax_localS0;
	char* h_lMax_localS0P1;
	char* h_lMax_localS1;      
	char* h_lMax_localS1P1;      

	int length = atoi(argv[1]);
	int NUM_THREADS = 10240;
	
	//cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	clock_t start,end;

	//Aloca memória dos vetores	
	//h_sequence = (char*) malloc(length);
	//h_threadSequences = (char*) malloc(length*NUM_THREADS);
	h_lMax_localS0 = (char*) malloc(NUM_THREADS);
	h_lMax_localS0P1 = (char*) malloc(NUM_THREADS);
	h_lMax_localS1 = (char*) malloc(NUM_THREADS);
	h_lMax_localS1P1 = (char*) malloc(NUM_THREADS);
	//cudaMalloc(&d_threadSequences, length*NUM_THREADS);
	cudaSetDevice(0);
	cudaMalloc(&d_lMax_localS0, NUM_THREADS);
	cudaMemset(d_lMax_localS0, 0, NUM_THREADS);
    cudaMalloc(&d_lMax_localS0P1, NUM_THREADS);
	cudaMemset(d_lMax_localS0P1, 0, NUM_THREADS);

	cudaSetDevice(1);
	cudaMalloc(&d_lMax_localS1, NUM_THREADS);
	cudaMemset(d_lMax_localS1, 0, NUM_THREADS);
	cudaMalloc(&d_lMax_localS1P1, NUM_THREADS);
	cudaMemset(d_lMax_localS1P1, 0, NUM_THREADS);


	start = clock();
	
	unsigned long long numSeq = fatorialHost(length-1)/2;

    int blockSize = 128;   // The launch configurator returned block size 
    int gridSize;    // The actual grid size needed, based on input size 

            // Round up according to array size 
    gridSize = ceil(NUM_THREADS / blockSize); 
	
	//dim3 num_blocks(ceil((float) NUM_THREADS/(float) (THREAD_PER_BLOCK)));
	//int tam_shared = length*THREAD_PER_BLOCK;
	int tam_shared = (length+1)*blockSize;
	printf("Começou\n");
	//Cada thread calcula: Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)), e se o resultado for maior que o máximo local,
	//insere na variável
	cudaSetDevice(0);
	//decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
	decideLS<<<gridSize, blockSize,  tam_shared>>>
	   (d_lMax_localS0, d_lMax_localS0P1,length, numSeq, NUM_THREADS, 0);	
	
	cudaSetDevice(1);
	//decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
	decideLS<<<gridSize, blockSize,  tam_shared>>>
	   (d_lMax_localS1, d_lMax_localS1P1, length, numSeq, NUM_DEVICES*NUM_THREADS, NUM_THREADS);

	cudaSetDevice(0);
	cudaMemcpyAsync(h_lMax_localS0, d_lMax_localS0, 1, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_lMax_localS0P1, d_lMax_localS0P1, 1, cudaMemcpyDeviceToHost);

	cudaSetDevice(1);
	cudaMemcpyAsync(h_lMax_localS1, d_lMax_localS1, 1, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_lMax_localS1P1, d_lMax_localS1P1, 1, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	char lMax_globalS = 0; //Variável com o máximo global de S
	if (h_lMax_localS0[0] < h_lMax_localS1[0]) {
		lMax_globalS = h_lMax_localS1[0];
	} else {
		lMax_globalS = h_lMax_localS0[0];
	}

	char lMax_globalSP1 = 0; //Variável com o máximo global de S
	if (h_lMax_localS0P1[0] < h_lMax_localS1P1[0]) {
		lMax_globalSP1 = h_lMax_localS1P1[0];
	} else {
		lMax_globalSP1 = h_lMax_localS0P1[0];
	}

	end = clock();

	printf("100%% - Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("%d Lmax R = %d\n",length, lMax_globalS);
	printf("%d Lmax R = %d\n",length+1, lMax_globalSP1);

	free(h_lMax_localS0);
	free(h_lMax_localS0P1);
	free(h_lMax_localS1);
	free(h_lMax_localS1P1);
	cudaFree(d_lMax_localS0);
	cudaFree(d_lMax_localS0P1);
	cudaFree(d_lMax_localS1);
	cudaFree(d_lMax_localS1P1);
}

