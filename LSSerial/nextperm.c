#include <stddef.h>
#include <stdio.h>
#include "LIS.c"
#include "LDS.c"
#include <time.h>
#include <string.h>
#include "EnumaratorSequence.c"


void rotation(char *array, int length){
  int i;
  char temp = array[0];
  for (i = 0; i < length-1; i++)
     array[i] = array[i+1];
  array[i] = temp;
}

int next_permutation(char *array, size_t length) {
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

void printVector(char* array, int length){
	int k;
	for(k = 0; k < length; k++){
		printf("%d - ",array[k]);	
	}
	printf("\n");
}

//Seja S o conjunto de todas las sequencias dos n primeiros números naturais.
//Defina R(s), com s \in S o conjunto de todas as sequencias que podem
//ser geradas rotacionando S.
//Defina LIS(s) e LDS(s) como você sabe e sejam |LIS(s)| e |LDS(s)| suas
//cardinalidades.
//Determinar Max_{s \in S}(Min_{s' \in R(s)}(Min(|LIS(s)|, |LDS(s)|)))
int main(int argc, char* argv[]){
	char* vector;
	char* vecRotation;
	
	int length = atoi(argv[1]);
	//	int lR_expected = atoi(argv[2]);
	int num_expected = 0;
	clock_t start,end;
	
	vector = (char*) malloc(length);
	vecRotation = (char*) malloc(length);

	start = clock();
	

	unsigned int lmaxS = 0;
	//Length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto.
	unsigned long long counter = fatorial(length-1)/2;
	int index = 1;
    //Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.

	unsigned int lLIS, lLDS;
	while(counter > index){
		counter--;
		getSequence(vector, length, index);

		unsigned int lminR = length;	
		
		lminR = LIS(vector, length);

		if(lminR <= lmaxS)
			continue;

		lLDS = LDS(vector, length);
		

		if(lLDS < lminR)
			lminR = lLDS;

		if(lLDS <= lmaxS)
			continue;

		memcpy(vecRotation,vector,length);
		for(i = 0; i < length-1; i++){
			rotation(vecRotation, length);

			lLIS = LIS(vecRotation, length);

			if(lLIS < lminR)
				lminR = lLIS;

			if(lLIS <= lmaxS)
				break;

			lLDS = LDS(vecRotation, length);
			
			if(lLDS < lminR)
				lminR = lLDS;

			if(lLDS <= lmaxS)
				break;			
		}
		
		//Define o maior valor encontrado entre os elementos de S
		if(lmaxS < lminR){
			lmaxS = lminR;
		}
		index++;
	}
	end = clock();
	
	printf("%d -> Tempo: %f s\n",length, (float)(end-start)/CLOCKS_PER_SEC);
	printf("Lmax S = %d\n",lmaxS);
	free(vector);
	free(vecRotation);
}
