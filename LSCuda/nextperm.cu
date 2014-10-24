#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
//#include "EnumaratorSequence.cu"
#include <time.h>

//#define NUM_THREADS 1024
#define THREAD_PER_BLOCK 1
#define N 10

__device__
void inversion(char* vet, int length){
	char temp;
	for(int i = 0; i < length; i++){
		temp = vet[length-i-1];
		vet[length-i-1] = vet[i];
		vet[i] = temp;
	}
	vet[length-1] = vet[0];
}

/* __device__
 void rotation(char* in, int length){
 	for(int i = 0; i < length-1; i++){
 		in[length+i] = in[i];
 	}
 }*/

__device__
void rotation(char *array, int length){
  char temp;
  int i;
  temp = array[0];
  for (i = 0; i < length-1; i++)
     array[i] = array[i+1];
  array[i] = temp;
}

int next_permutation(char *array, size_t length) {
	size_t i, j;
	char temp;
	// Find non-increasing suffix
	if (length == 0)
		return 0;
	i = length - 1;
	while (i > 0 && array[i - 1] >= array[i])
		i--;
	if (i == 0)
		return 0;
	
	// Find successor to pivot
	j = length - 1;
	while (array[j] <= array[i - 1])
		j--;
	temp = array[i - 1];
	array[i - 1] = array[j];
	array[j] = temp;
	
	// Reverse suffix
	j = length - 1;
	while (i < j) {
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
		i++;
		j--;
	}
	return 1;
}

__device__
void printVector(char* array, int length){
	for(int k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

unsigned long fatorial(unsigned long n){
	int i;
	unsigned long result = 1;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

//Min(|LIS(s)|, |LDS(s)|)
__global__
void decideLS(char *vector, char* d_lMax_S, int length, int numThread, int step_seq){
	//Step_shared - quantidade de posições utilizada por cada thread
	//Step_seq - quantidade de posições utilizadas pela sequência
	//step_last - quantidade de posições utilizado pelo vetor Lasto do LSI/LDS
	extern __shared__ char s_vet[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 	
	int s_index = step_seq*threadIdx.x; //Indice da shared memory
	if(tid < numThread){
		
		for(int i = 0; i < step_seq; i++){
			s_vet[s_index+i] = vector[tid*step_seq+i];
		}
		
		char MP[N*(N+1)/2];
		char last[N];

		char lLIS, lLDS; 
		char lMin_R = 127;

		for(int j = 0; j < 2; j++){ //Inverção
			for(int i = 0; i < length; i++){
				
				lLIS = LIS(s_vet + s_index, last, MP, length);
				
				if(lLIS < lMin_R){
					printf("lLIS: %d\n", lLIS);
					printVector(s_vet + s_index, length);
					lMin_R = lLIS;	
				}

				//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
				if(lLIS <= d_lMax_S[tid])
					return;				

				lLDS = LDS(s_vet + s_index, last, MP, length);
				if(lLDS < lMin_R){
					lMin_R = lLDS;
				}

				//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
				if(lLDS <= d_lMax_S[tid])
					return;

				rotation(s_vet + s_index, length);
			}

			//Não fazer a inverção duas vezes. PENSAR EM METODO MELHOR
			if(j == 1)
				return;
			else{
				inversion(s_vet + s_index, length);
			}
		}
		printf("%d\n,lMin_R");
		d_lMax_S[tid] = lMin_R;
	}
}

void calcLMaxS(char* lMax_S, char* lMin_R, int tamVec){
	//Número de conjuntos
	for(int i = 0; i < tamVec; i++){
		if(*lMax_S < lMin_R[i]){
			*lMax_S = lMin_R[i];
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
	char* h_sequence;            //Vetor com a sequência pivor do grupo
	char* h_threadSequences;      //Vetor com as sequências criadas
	char* d_threadSequences;	    //Sequências produzidas para enviar para o device
	char* d_lMax_S;      //Vetor com os resultados de cada thread. L Mínimos do conjunto de R
	char* h_lMax_S;      

	int length = N;
	int NUM_THREADS = atoi(argv[1]);
	

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	//Tamanho linear da sequência que vai ser enviada para cada thread.
	//Vetor consisti em Sua sequência seguida por repetição dos seus primeiros length-1 elementos devido a rotação.
	//Ex: 1 2 3 4 1 2 3.
	//int step_element = 2*length-1; 
	int step_element = length; 

	//Tamanho linear de cada thread da Shared Memory, composto por:
	//Vetor MR[N+1]*[N+1] com as sequência LIS e LDS mais promissoras
	//Vetor Last[N] com o tamanho do ultimo valor de cada sequência promissora
	//Sequência geradas de tamanho step_element
	printf("Shared: %d\n", step_element*THREAD_PER_BLOCK);
	clock_t start,end;

	//Aloca memória dos vetores	
	h_sequence = (char*) malloc(length);
	h_threadSequences = (char*) malloc(step_element*NUM_THREADS);
	h_lMax_S = (char*) malloc(NUM_THREADS);
	cudaMalloc(&d_threadSequences, step_element*NUM_THREADS);
	cudaMalloc(&d_lMax_S, NUM_THREADS);

	//Gera a sequência primária, de menor ordem léxica	
	for(int i = 0; i < length; i++)
		h_sequence[i] = i+1;

	unsigned int numSeqReady = 0; //Número de sequêcias prontas
	//char lMax_S = 0; //Resultado final, maior valor encontrado do grupo S

	start = clock();
	
	next_permutation(h_sequence+1,length-1); //Remover a primeira sequência, pois o resultado é sempre 1

	//length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto. E -1 devido a remoção da primeira sequência
	unsigned long counter = fatorial(length-1)/2 -1;
	unsigned long counterMax = counter;
	//Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
	while(counter){
		
		//Gera todos os pivores do conjunto R
		memcpy(h_threadSequences + numSeqReady*step_element,
			   h_sequence, length);
		numSeqReady++;
		
		//Caso não tenha como inserir mais un conjunto inteiro no número de threads, então executa:
		if(numSeqReady == NUM_THREADS){
			cudaMemcpy(d_threadSequences, h_threadSequences, numSeqReady*step_element, cudaMemcpyHostToDevice);
			
			dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
			int tam_shared = step_element*THREAD_PER_BLOCK;
			//Cada thread calcula: Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|))
			decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
					   (d_threadSequences, d_lMax_S, length, numSeqReady, step_element);
					
			//Recomeça a gerar sequências
			numSeqReady = 0; 
		}	

		//Cria a próxima sequência na ordem lexicográfica
		next_permutation(h_sequence+1,length-1);
		counter--;

		if((counterMax - counter)%(counterMax/100) == 0){
			end = clock();
			printf("%lu%% - Tempo: %f s - Counter: %lu\n",((counterMax - counter)/(counterMax/100)), (float)(end-start)/CLOCKS_PER_SEC, counter);
		}
	}

	//Calculo do Resto, que foi gerado, porèm não encheu o vetor de sequências geradas.
	if(numSeqReady != 0){
		cudaMemcpy(d_threadSequences, h_threadSequences, numSeqReady*(2*length-1), cudaMemcpyHostToDevice);
			
		dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
		int tam_shared = ((length+1)*(length+1)+(3*length-1))*THREAD_PER_BLOCK*sizeof(int);
		
		decideLS<<<num_blocks,THREAD_PER_BLOCK, tam_shared>>>
			       (d_threadSequences, d_lMax_S, length, numSeqReady, step_element);
		
	}

	cudaMemcpy(h_lMax_S, d_lMax_S, numSeqReady, cudaMemcpyDeviceToHost);
	char lMax_S = 0;
	calcLMaxS(&lMax_S, h_lMax_S, numSeqReady);	

	cudaThreadSynchronize();
	end = clock();

	printf("100%% - Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("Lmax R = %d\n",lMax_S);

	free(h_sequence);
	free(h_threadSequences);
	free(h_lMax_S);
	cudaFree(d_threadSequences);
	cudaFree(d_lMax_S);
}
