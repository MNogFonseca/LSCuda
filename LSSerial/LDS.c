#define STACKSIZE 32768    /* tamanho de pilha das threads */
#define _XOPEN_SOURCE 600  /* para compilar no MacOS */


#include <stdio.h>
#include <stdlib.h>
#define N 4

//pega o menor valor do vetor last que seja maior do que x
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
void LDSVetCopy(int* dest, int* in,int tam){
	int i;
	for(i = 0; i<tam;i++){
		dest[i] = in[i];
	}
}

//printa a matriz de mais provaveis sequencias de serem a LIS
void LDSprintMP(int* mat,int tam){
	int i;
	for(i =0 ; i<tam ; i++){

		if(mat[i]== -1){
			break;
		}
		else
			printf("%d -",mat[i]);
	}
	printf("\n");	
}



unsigned int LDS(int* vet, int tam){	
	int *last = (int*) malloc(sizeof(int)*tam); //inicializa o vetor com
						//com os ultimos elementos de MP
	int i;											 
	for(i =0;i<tam;i++){
		last[i] = 0;
	}
	
	int lmax = 1;  //maior tamanho de subsequencia

	int** MP = (int**) malloc(sizeof(int*)*(tam+1)); //inicializa a matriz de mais promissores
	for(i = 0; i < tam+1; i++){
		MP[i] = (int*) malloc(sizeof(int)*tam);
	}

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
	
	return lmax;
}
/*
int main(){
	int vet[N] = {9,5,3,4};
	int i;
	

	int lmax = LDS(vet,N);
	printf("lmax = %d\n", lmax);

	return 0;
}*/
