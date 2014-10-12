#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
#include <time.h>

#define NUM_THREADS 512
#define THREAD_PER_BLOCK 64
/*
#define NUM_SM 8
#define MAX_THREAD_PER_SM 2048
#define length 10
#define MAX_SHARED_PER_BLOCK 49152
#define SHARED_PER_THREAD 	(length*length+length)
#define THREAD_PER_BLOCK 	MAX_SHARED_PER_BLOCK/SHARED_PER_THREAD
#define NUM_BLOCKS 			(THREAD_PER_SM*NUM_SM)/THREAD_PER_BLOCK
#define NUM_THREADS 		NUM_BLOCKS*THREAD_PER_BLOCK
*/

__device__
void inversion(int* dest, int* in, int length){
	for(int i = 0; i < length; i++){
		dest[i] = in[length-i-1];
	}
}
/*
void rotation(int* dest, int* in, int length){
  int i;	
  dest[0] = in[length-1];
  for (i = 1; i < length; i++)
     dest[i] = in[i-1];
}*/

 __device__
 void rotation(int* in, int length){
 	for(int i = 0; i < length-1; i++){
 		in[length+i] = in[i];
 	}
 }

int next_permutation(int *array, size_t length) {
	size_t i, j;
	int temp;
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

void printVector(int* array, int length){
	for(int k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

int fatorial(int n){
	int result = 1;
	for(int i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

void criaSequencias(int* dest, int* in,int length, unsigned int* numSeqReady){
	//Inserir o pivor em primeiro lugar com suas rotações, e sua inversão também com suas rotações	
	memcpy(dest,in, sizeof(int)*length);
	memcpy(dest+length,in, sizeof(int)*(length-1));

	/*inversion(dest+(2*length-1), dest, length);
	memcpy(dest+(3*length-1),dest+(2*length-1), sizeof(int)*(length-1));
	*numSeqReady += 2;*/
	(*numSeqReady)++;
	
	/*
	//Rotaciona o pivor, e inverte os elementos produzidos
	int i;
	for(i = 1; i < (length); i++, *numSeqReady+=2){
		rotation(dest + (2*i)*length,dest + (2*i-2)*length, length); //Diminuição de dois elementos, para pular a inversão do pivor
		inversion(dest + (2*i+1)*length,dest+(2*i)*length, length);		
	}*/
}

//Min(|LIS(s)|, |LDS(s)|)
__global__
void decideLS(int *vector, unsigned int* lmin, int length, int numThread, int lMax_S){
	extern __shared__ int s_vet[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x; 
	int s_step = (length+1)*(length+1) + 3*length -1;
	int s_index = s_step*threadIdx.x; //Indice da shared memory
	if(tid < numThread){
		
		for(int i = 0; i < (2*length-1); i++){
			s_vet[s_index+i] = vector[tid*(2*length-1)+i];
		}

		unsigned int lLIS, lLDS; 
		lmin[tid] = 1000;

		for(int j = 0; j < 2; j++){ //Inverção
			for(int i = 0; i < length; i++){
				lLIS = LIS(s_vet + s_index + i, s_vet + s_index + (2*length-1), s_vet + s_index + (3*length-1), length);
				if(lLIS < lmin[tid]){
					lmin[tid] = lLIS;	
				}

				if(lLIS < lMax_S)
					return;

				lLDS = LDS(s_vet + s_index + i, s_vet + s_index + (2*length-1), s_vet + s_index + (3*length-1), length);;	
				if(lLDS < lmin[tid]){
					lmin[tid] = lLDS;
				}

				if(lLDS < lMax_S)
					return;
			}
			if(tid == 0){
				for(int i = 0; i < 2*length-1; i++){
					printf("%d - ", s_vet[s_index+i]);
				}
				printf("\n");
			}
			inversion(s_vet + s_index, s_vet + s_index + length -1, length);
			rotation(s_vet + s_index, length);
			if(tid == 0){
				for(int i = 0; i < 2*length-1; i++){
					printf("%d - ", s_vet[s_index+i]);
				}
				printf("\n\n");
			}
		}
	}
	
}

int reduceLMinR(unsigned int* lMin_s, int tam){
	unsigned int lMin_R = 0xFF;
	for(int i = 0; i < tam; i++){
		//printf("RLM - %d - %d\n",i, lMin_s[i]);
		if(lMin_R > lMin_s[i]){
			lMin_R = lMin_s[i];	
		}
	}
	return lMin_R;
}

void calcLMaxS(unsigned int* lMax_S, unsigned int* lMin_s, int tamVec, int tamGroup){
	unsigned int lMin_R;
	//Número de conjuntos
	for(int i = 0; i < tamVec/tamGroup; i++){
		lMin_R = reduceLMinR(lMin_s+i*tamGroup, tamGroup);
		
		if(*lMax_S < lMin_R){
			*lMax_S = lMin_R;
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
	int* h_sequence;            //Vetor com a sequência pivor do grupo
	int* h_threadSequences;      //Vetor com as sequências criadas
	int* d_threadSequences;	    //Sequências produzidas para enviar para o device
	unsigned int* d_lMin_s;      //Vetor com os resultados de cada thread. L Mínimos do conjunto de R
	unsigned int* h_lMin_s;      

	int length = atoi(argv[1]);

	clock_t start,end;

	//Aloca memória dos vetores	
	h_sequence = (int*) malloc(sizeof(int)*length);
	h_threadSequences = (int*) malloc(sizeof(int)*(2*length-1)*NUM_THREADS);
	h_lMin_s = (unsigned int*) malloc(sizeof(unsigned int)*NUM_THREADS);
	cudaMalloc(&d_threadSequences, sizeof(int)*(2*length-1)*NUM_THREADS);
	cudaMalloc(&d_lMin_s, sizeof(int)*NUM_THREADS);

	//Gera a sequencia primária, de menor ordem léxica	
	for(int i = 0; i < length; i++)
		h_sequence[i] = i+1;

	unsigned int numSeqReady = 0; //Número de sequêcias prontas

	start = clock();
	unsigned int lMax_S = 0;

	//length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto.
	int counter = fatorial(length-1)/2;
        
    //Número de elementos em cada conjunto. length (rotação) * 2 (inversão)    
	int tamGroup = 1;

	//Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
	while(counter){
		
		//Gera todo o conjunto R
		criaSequencias(h_threadSequences + numSeqReady*(2*length-1), //Vetor com as sequências geradas
		    		   h_sequence, //Vetor pivor
                       length,
			           &numSeqReady); //Número de threads prontos
		
		//Caso não tenha como inserir mais un conjunto inteiro no número de threads, então executa:
		if((numSeqReady+tamGroup) >= NUM_THREADS){
			cudaMemcpy(d_threadSequences, h_threadSequences, sizeof(int)*numSeqReady*(2*length-1), cudaMemcpyHostToDevice);
			
			
			//Cada thread calcula o LIS e o LDS de cada sequência
			dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
			int tam_shared = ((length+1)*(length+1)+3*length-1)*THREAD_PER_BLOCK*sizeof(int);
			
			decideLS<<<num_blocks, THREAD_PER_BLOCK,  tam_shared>>>
					   (d_threadSequences, d_lMin_s, length, numSeqReady, lMax_S);
					
			cudaMemcpy(h_lMin_s, d_lMin_s, sizeof(unsigned int)*numSeqReady, cudaMemcpyDeviceToHost);

			calcLMaxS(&lMax_S, h_lMin_s, numSeqReady, tamGroup);	
			numSeqReady = 0; 
		}	

		//Cria a próxima sequência na ordem lexicográfica
		next_permutation(h_sequence+1,length-1);
		counter--;
	}

	if(numSeqReady != 0){
		cudaMemcpy(d_threadSequences, h_threadSequences, sizeof(int)*numSeqReady*(2*length-1), cudaMemcpyHostToDevice);
			
		//Cada thread calcula o LIS e o LDS de cada sequência
		dim3 num_blocks(ceil(((float) numSeqReady)/(float) THREAD_PER_BLOCK));
		int tam_shared = ((length+1)*(length+1)+(3*length-1))*THREAD_PER_BLOCK*sizeof(int);
		
		decideLS<<<num_blocks,THREAD_PER_BLOCK, tam_shared>>>
			       (d_threadSequences, d_lMin_s, length, numSeqReady, lMax_S);
		
		cudaMemcpy(h_lMin_s, d_lMin_s, sizeof(unsigned int)*numSeqReady, cudaMemcpyDeviceToHost);

		calcLMaxS(&lMax_S, h_lMin_s, numSeqReady, tamGroup);	
	}

	cudaThreadSynchronize();
	end = clock();

	printf("Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("Lmax R = %d\n",lMax_S);

	free(h_sequence);
	free(h_threadSequences);
	free(h_lMin_s);
	cudaFree(d_threadSequences);
	cudaFree(d_lMin_s);
}
