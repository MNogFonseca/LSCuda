#include <stdio.h>
#include <stdlib.h>
#if !defined LENGTH
#define LENGTH 5
#endif
//pega o menor valor do vetor last que seja maior do que x
__device__
int LDSgetLast(int* last,int x,int tam){
	int i;
	for(i=0; i < tam ;i++){
		if(last[i] < x ){
			return i+1;
		}		
	}

	return -1;
}

//pega a posicao valida para inserir um elemento no vetor vet
__device__
int LDSgetPos(int vet[],int tam){
	int i;
	for(i =0;i < tam;i++){

		if(vet[i]== -1){
			return i;
		}
	}
	return -1;
}

//copia um vetor para outro
__device__
void LDSVetCopy(int* dest, int* in,int tam){
	int i;
	for(i = 0; i<tam;i++){
		dest[i] = in[i];
	}
}

__device__
unsigned int LDS(int* vet, int tam){	
	int last[LENGTH]; //inicializa o vetor com
						//com os ultimos elementos de MP
	int i;											 
	for(i =0;i<tam;i++){
		last[i] = 0;
	}
	
	int lmax = 1;  //maior tamanho de subsequencia

	int MP[LENGTH+1][LENGTH+1]; //inicializa a matriz de mais promissores

	for(i = 0;i<tam; i++){
		int j;
		for(j = 0;j<tam; j++){
			
			MP[i][j] = -1;
		}
	}


	MP[1][0] = vet[0];
	last[0] = vet[0];

	for(i=1; i < tam; i++){

		int l = LDSgetLast(last,vet[i],tam); //pega  valor de l

		//atualiza o valor de lmax
		if(l > lmax){ 
			lmax ++;
		}

			last[l-1] = vet[i]; //atualiza o vetor last

		 	//concatena os vetores de MP
			LDSVetCopy(MP[l],MP[l-1],tam);

			int pos = LDSgetPos(MP[l],tam);			
			MP[l][pos] = last[l-1];	
	}
	
	for(i = 0; i < tam+1; i++){
		free(MP[i]);
	}
	free(MP);
	free(last);
	return lmax;
}