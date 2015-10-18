#include <stddef.h>
#include <stdio.h>
#include <math.h>
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
	
	vector = (char*) malloc(length+1);
	vecRotation = (char*) malloc(length+1);

	unsigned int lmaxS = 0;
	//Length -1 porque devido a rotação pode sempre deixar o primeiro número fixo, e alternar os seguintes
	//Dividido por 2, porque a inversão cobre metade do conjunto.
	unsigned long long index = 0;
    //Cada loop gera um conjunto de sequências. Elementos de S. Cada elemento possui um conjunto de R sequencias.
    FILE* file = fopen("ind", "r");
    fscanf (file, "%llu", &index);
	unsigned int lLIS, lLDS;
	while(!feof (file)) {
		getSequence(vector, length, index);
        vector[length] = length+1;
        printVector(vector, length+1);

		unsigned int lminR = length+1;	
		
		lminR = LIS(vector, length+1);

		lLDS = LDS(vector, length+1);
		if(lLDS < lminR)
			lminR = lLDS;

		memcpy(vecRotation,vector,length+1);
		int i;
		for(i = 0; i < length; i++){
			rotation(vecRotation, length+1);

			lLIS = LIS(vecRotation, length+1);

			if(lLIS < lminR)
				lminR = lLIS;

			lLDS = LDS(vecRotation, length+1);
			
			if(lLDS < lminR)
				lminR = lLDS;
		}
		if (lminR == ceil((length+1)/3.0)) {
            printf ("%llu\n", index);
        //printVector(vector, length);

        }
		//Define o maior valor encontrado entre os elementos de S
		if(lmaxS < lminR){
			lmaxS = lminR;
		}
        fscanf (file, "%llu", &index);
	}
	//printf("%d -> Tempo: %f s\n",length, (float)(end-start)/CLOCKS_PER_SEC);
	//printf("Lmax S = %d\n",lmaxS);
    //printf ("\n%d == ", num);
	free(vector);
	free(vecRotation);
}
