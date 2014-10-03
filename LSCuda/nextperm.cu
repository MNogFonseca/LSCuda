#include <stddef.h>
#include <stdio.h>
#include "LIS.cu"
#include "LDS.cu"
#include <time.h>
#define NUM_THREADS 1024
#define THREAD_BLOCK 512

void inversion(int* dest, int* in, int length){
	int i;
	for(i = 0; i < length; i++){
		dest[i] = in[length-i-1];
	}
}

void rotation(int* dest, int* in, int length){
  int i;	
  dest[0] = in[length-1];
  for (i = 1; i < length; i++)
     dest[i] = in[i-1];

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
	int k;
	for(k = 0; k < 4; k++){
		//printf("%d - ",array[k]);	
	}
	//printf("\n");
}

int fatorial(int n){
	int i;
	int result = 1;
	for(i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

int criaSequencias(int* dest, int* in,int length, int posInicial, unsigned int* numSec){
	int i;

	if(posInicial == 0){
		inversion(dest+length, dest,length);
	}
	else{
		rotation(dest, in, length);
		inversion(dest+length, dest,length);
	}

	*numSec = 2;
	posInicial++;
	for(i = posInicial; i < (length); i++, *numSec+=2){
		if(*numSec == NUM_THREADS){
			return i;		
		} 

		rotation(dest + *numSec*length,dest + (*numSec-2)*length, length);
		inversion(dest + (*numSec+1)*length,dest+(*numSec)*length, length);
		
	}
	return -1;
}

//Min(|LIS(s)|, |LDS(s)|)
__global__
void decideLS(int *vector, unsigned int* lmin, int length, int numThread){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index <= numThread){
		unsigned int lLIS, lLDS; 
	
		lLIS = LIS(vector+index, length);
		lLDS = LDS(vector+index, length);

		if(lLIS < lmin[index]){
			lmin[index] = lLIS;
		}

		if(lLDS < lmin[index]){
			lmin[index] = lLDS;	
		}

	}
	
}

void reduceLMinR(unsigned int* lmin_R, unsigned int* h_lMin_s, int tam){
	int i;
	for(i = 0; i < tam; i++){
		if(*lmin_R > h_lMin_s[i]){
			*lmin_R = h_lMin_s[i];	
		}
	}

}
//Seja S o conjunto de todas las sequencias dos n primeiros números naturais.
//Defina R(s), com s \in S o conjunto de todas as sequencias que podem
//ser geradas rotacionando S.
//Defina LIS(s) e LDS(s) como você sabe e sejam |LIS(s)| e |LDS(s)| suas
//cardinalidades.
//Determinar Max_{s \in S}(Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)))


int main(){
	int* h_sequence;            //Vetor com a sequência pivor do grupo
	int* h_threadSequences;      //Vetor com as sequências criadas
	int* d_threadSequences;	    //Sequências produzidas para enviar para o device
	unsigned int* d_lMin_s;      //Vetor com os resultados de cada thread. L Mínimos do conjunto de R
	unsigned int* h_lMin_s;      
	int length = 4;
	clock_t start,end;

	//Aloca memória dos vetores	
	h_sequence = (int*) malloc(sizeof(int)*length);
	h_threadSequences = (int*) malloc(sizeof(int)*length*NUM_THREADS);
	h_lMin_s = (unsigned int*) malloc(sizeof(unsigned int)*NUM_THREADS);
	cudaMalloc(&d_threadSequences, sizeof(int)*length*NUM_THREADS);
	cudaMalloc(&d_lMin_s, sizeof(int)*NUM_THREADS);

	//Gera a sequencia primária, de menor ordem léxica	
	int i;
	for(i = 0; i < length; i++)
		h_sequence[i] = i+1;

	unsigned int numSeqReady = 0;

	start = clock();
	unsigned int lMax_S = 0;

	//Length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto.
	int counter = fatorial(length-1)/2;
        //Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
	while(counter){
		int posInicial = 0;

		memcpy(h_threadSequences,h_sequence, sizeof(int)*length);
		cudaMemset(d_lMin_s, 0xFF, sizeof(unsigned 	int)*NUM_THREADS); //Seta os vetor com um número muito grande			
		unsigned int lMin_R = 0xFF;
		while(1){
			posInicial = criaSequencias(h_threadSequences, //Vetor com as sequências geradas
						    		    h_threadSequences+(NUM_THREADS-2)*length, //Caso posInical !=1, esse ponteiro tem o ultimo elemento calculado sem ser inversão
                                        length, posInicial, //Tamanho do Elemento, e quantidade de posições ja calculadas 
				                    	&numSeqReady); //Número de threads prontos
			
			cudaMemcpy(d_threadSequences, h_threadSequences, sizeof(int)*NUM_THREADS*length, cudaMemcpyHostToDevice);
			cudaThreadSynchronize();
			decideLS<<<numSeqReady%THREAD_BLOCK, ceil(((float) numSeqReady)/(float) THREAD_BLOCK)>>>(d_threadSequences, d_lMin_s, length, numSeqReady);
			cudaThreadSynchronize();
			cudaMemcpy(h_lMin_s, d_lMin_s, sizeof(unsigned int)*NUM_THREADS, cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();	
			reduceLMinR(&lMin_R, h_lMin_s, numSeqReady); //Calcular o lMinR das sequências ja calculadas

			//Todos elementos do conjunto R já foram gerados
			if(posInicial == -1){
				break;
			}
			cudaMemset(d_lMin_s, 0xFF, sizeof(unsigned 	int)*NUM_THREADS); //Seta os vetor com um número muito grande			
		}

		//Define o maior valor encontrado entre os elementos de S
		if(lMax_S < lMin_R){
			lMax_S = lMin_R;
		}

		//Cria a próxima sequência na ordem lexicográfica
		next_permutation(h_sequence+1,length-1);
		//printf("\n");
		counter--;
	}
	end = clock();

	printf("Tempo: %f s\n", (float)(end-start)/CLOCKS_PER_SEC);

	printf("Lmax R = %d\n",lMax_S);
}
