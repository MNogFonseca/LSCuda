#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
//#include "EnumaratorSequence.cu"
#include <time.h>

//#define NUM_THREADS 1024
#define THREAD_PER_BLOCK 128
#define N 16

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

//faz a proxima permutação na ordem lexicográfica
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

unsigned long fatorial(unsigned long n){
	int i;
	unsigned long result = 1;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

//Calcula o LIS de todo o conjunto R partindo do pivor principal da ordem lexico gráfica
//Caso encontre um valor que é menor do que o máximo local de S, então ele retorna e não faz os outros calculos.
__global__
void decideLS(char *vector, char* d_lMax_S, int length, int numThread){
	extern __shared__ char s_vet[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 	
	int s_index = length*threadIdx.x; //Indice da shared memory

	if(tid < numThread){
		//carrega o vetor na shared memory
		for(int i = 0; i < length; i++){
			s_vet[s_index+i] = vector[tid*length+i];
		}
		
		//Esses dois vetores são utilizados no LIS e no LDS, são declarados do lado de fora para
		//gastar menos memória e não ter necessidade de dar malloc.
		char MP[N*(N+1)/2]; //Vetor de most promising
		char last[N]; //Vetor de last de MP

		//Valores com os resultados encontrados no LIS e no LDS
		char lLIS, lLDS; 

		char lMin_R = 127; //Variavel que representa o min encontrado no conjunto R

		for(int i = 0; i < length; i++){ //Rotação

			lLIS = LIS(s_vet + s_index, last, MP, length);
			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLIS < lMin_R){
				lMin_R = lLIS;	
			}

			//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
			if(lLIS <= d_lMax_S[tid]){
				return;				
			}

			lLDS = LDS(s_vet + s_index, last, MP, length);
			//caso seja menor que o minimo do conjunto R, então modificar o valor
			if(lLDS < lMin_R){				
				lMin_R = lLDS;
			}

			//Todo o conjunto pode ser descartado, pois não vai subistituir lMax_S no resultado final
			if(lLDS <= d_lMax_S[tid]){
				return;
			}

			rotation(s_vet + s_index, length);
		}
		//Caso o resultado final encontrado de R chegue ate o final, então significa que ele é maior
		//Que o minimo local encontrado até o momento.
		if(lMin_R == 6){
			printVector(s_vet+s_index, length);
		}
		d_lMax_S[tid] = lMin_R;		
	}
}

//Com os valores de máximos locais de S, calcular o máximo global.
void calcLMaxGlobalS(char* lMax_globalS, char* lMax_localS, int tamVec){
	//Número de conjuntos
	for(int i = 0; i < tamVec; i++){
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
	char* h_sequence;            //Vetor com a sequência pivor do grupo
	char* h_threadSequences;      //Vetor com as sequências criadas
	char* d_threadSequences;	    //Sequências produzidas para enviar para o device
	char* d_lMax_localS;      //Vetor com os máximos locais de S, cada thread tem um máximo local
	char* h_lMax_localS;      

	int length = atoi(argv[1]);
	int NUM_THREADS = atoi(argv[2]);
	
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	clock_t start,end;

	//Aloca memória dos vetores	
	h_sequence = (char*) malloc(length);
	h_threadSequences = (char*) malloc(length*NUM_THREADS);
	h_lMax_localS = (char*) malloc(NUM_THREADS);
	cudaMalloc(&d_threadSequences, length*NUM_THREADS);
	cudaMalloc(&d_lMax_localS, NUM_THREADS);
	cudaMemset(d_lMax_localS, 0, NUM_THREADS);

	//Gera a sequência primária, de menor ordem léxica	
	for(int i = 0; i < length; i++)
		h_sequence[i] = i+1;
	unsigned int numSeqReady = 0; //Número de sequêcias prontas para enviar para a GPU

	start = clock();
	
	next_permutation(h_sequence+1,length-1); //Remover a primeira sequência, pois o resultado é sempre 1

	//length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto. E -1 devido a remoção da primeira sequência
	unsigned long counter = fatorial(length-1)/2 -1;
	unsigned long counterMax = counter;
	//Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
	while(counter){
		//Gera todos os pivores do conjunto R
		memcpy(h_threadSequences + numSeqReady*length, h_sequence, length);
		numSeqReady++;

		//Caso não tenha como inserir mais un conjunto inteiro no número de threads, então executa:
		if(numSeqReady == NUM_THREADS){
			cudaMemcpy(d_threadSequences, h_threadSequences, numSeqReady*length, cudaMemcpyHostToDevice);
			
			dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
			int tam_shared = length*THREAD_PER_BLOCK;

			//Cada thread calcula: Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)), e se o resultado for maior que o máximo local,
			//insere na variável
			decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
					   (d_threadSequences, d_lMax_localS, length, numSeqReady);
			//Recomeça a gerar sequências
			numSeqReady = 0; 
		}	
		//Cria a próxima sequência na ordem lexicográfica
		next_permutation(h_sequence+1,length-1);
		counter--;
		
		/*if((counterMax - counter)%(counterMax/100+1) == 0){
			end = clock();
			printf("%lu%% - Tempo: %f s - Counter: %lu\n",((counterMax - counter)/(counterMax/100+1)), (float)(end-start)/CLOCKS_PER_SEC, counter);
		}*/
	}

	//Calculo do Resto, que foi gerado, porèm não encheu o vetor de sequências geradas.
	if(numSeqReady != 0){
		cudaMemcpy(d_threadSequences, h_threadSequences, numSeqReady*(2*length-1), cudaMemcpyHostToDevice);
			
		dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
		int tam_shared = ((length+1)*(length+1)+(3*length-1))*THREAD_PER_BLOCK*sizeof(int);
		
		decideLS<<<num_blocks,THREAD_PER_BLOCK, tam_shared>>>
			       (d_threadSequences, d_lMax_localS, length, numSeqReady);
		
	}

	cudaMemcpy(h_lMax_localS, d_lMax_localS, NUM_THREADS, cudaMemcpyDeviceToHost);

	char lMax_globalS = 0; //Variável com o máximo global de S
	calcLMaxGlobalS(&lMax_globalS, h_lMax_localS, NUM_THREADS);	

	cudaThreadSynchronize();
	end = clock();

	printf("100%% - Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("Lmax R = %d\n",lMax_globalS);

	free(h_sequence);
	free(h_threadSequences);
	free(h_lMax_localS);
	cudaFree(d_threadSequences);
	cudaFree(d_lMax_localS);
}
