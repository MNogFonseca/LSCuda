#include <stdio.h>
#include <stdlib.h>
//#define MAP_MP(X) (X*(X-1))/2

//pega o menor valor do vetor last que seja maior do que x
int LISgetLast(char* last,int x,int tam){
	int i;
	for(i=0; i < tam ;i++){
		if(last[i] > x ){
			return i+1;
		}		
	}

	return -1;
}

//pega a posicao valida para inserir um elemento no vetor vet
int LISgetPos(char vet[],int tam){
	int i;
	for(i =0;i < tam;i++){
		if(vet[i]== -1){
			return i;
		}
	}
	return -1;
}

//copia um vetor para outro
void LISVetCopy(char* dest, char* in,int tam){
	int i;
	for(i = 0; i<tam-1;i++){
		dest[i] = in[i];
	}
	dest[tam-1] = -1;
}

/*
//printa a matriz de mais provaveis sequencias de serem a LIS
void LISprintMP(char* mat,int tam){
	int i, j;
	int pos = 0;
	for(i = 1 ; i<tam+1 ; i++){
		for(j = 0; j < i; j++, pos++){ 
			printf("%d |",mat[pos]);
		}
		printf("\n");	
		//if(mat[i]== -1){
		//	break;
		//}
		//else
		//	printf("%d -",mat[i]);

	}
	printf("\n");	
}
*/
int MAP_MP(int n){
	return n*(n-1)/2;
}

unsigned int LIS(char* vet, int tam){

	char* last = (char*) malloc(tam); //inicializa o vetor com
						//com os ultimos elementos de MP
	int i;											 
	for(i =0;i<tam;i++){
		last[i] = 127;
	}
	
	int lmax = 1;  //maior tamanho de subsequencia


	char* MP = (char*) malloc(MAP_MP(tam+1)); //inicializa a matriz de mais promissores
	for(i = 0;i< MAP_MP(tam+1); i++){
		MP[i] = -1;
	}



	MP[0] = vet[0];
	last[0] = vet[0];

	for(i=1; i < tam; i++){

		int l = LISgetLast(last,vet[i],tam); //pega  valor de l

		//atualiza o valor de lmax
		if(l > lmax){ 
			lmax ++;
		}

			last[l-1] = vet[i]; //atualiza o vetor last

		 	//concatena os vetores de MP
			LISVetCopy(MP+MAP_MP(l),MP+MAP_MP(l-1),l);

			int pos = LISgetPos(MP+MAP_MP(l),tam);			
			MP[MAP_MP(l)+pos] = last[l-1];
			//LISprintMP(MP, tam);
	}

	free(last);
	free(MP);
	return lmax;
}

/*#define N 16
int main(){
	int vet[N] = {16, 5, 10, 15, 4, 11, 12, 13, 1, 9, 8, 2, 14, 3, 7, 6};
	int i;

	int lmax = LIS(vet,N);
	printf("lmax = %d\n", lmax);

	return 0;
}
*/