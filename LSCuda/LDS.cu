#include <stdio.h>
#include <stdlib.h>
#define MAP_MP(X) (X*(X-1))/2
//pega o menor valor do vetor last que seja maior do que x
__device__
char LDSgetLast(char* last,int x,int tam){
	int i;
	for(i=0; i < tam ;i++){
		if(last[i] < x ){
			return i+1;
		}		
	}
	//Nunca deve chegar aqui
	return 0;
}

//pega a posicao valida para inserir um elemento no vetor vet
__device__
int LDSgetPos(char vet[],int tam){
	for(int i =0;i < tam;i++){

		if(vet[i]== -1){
			return i;
		}
	}
	return -1;
}

//copia um vetor para outro
__device__
void LDSVetCopy(char* dest, char* in,int tam){
	for(int i = 0; i<tam-1;i++){
		dest[i] = in[i];
	}
	dest[tam-1] = -1;
}

__device__
unsigned char LDS(char* vet, char* last, char* MP, int tam){

	//inicializa o vetor com os ultimos elementos de MP
	for(int i =0;i<tam;i++){
		last[i] = 127;
	}
	
	char lmax = 1;  //maior tamanho de subsequencia
	
	for(int i = 0;i<tam; i++){
		int j;
		for(j = 0;j<tam; j++){
			
			MP[i*tam+j] = -1;
		}
	}

	MP[0] = vet[0];
	last[0] = vet[0];

	for(int i=1; i < tam; i++){

		int l = LDSgetLast(last,vet[i],tam); //pega  valor de l

		//atualiza o valor de lmax
		if(l > lmax){ 
			lmax ++;
		}

			last[l-1] = vet[i]; //atualiza o vetor last

		 	//concatena os vetores de MP
			LDSVetCopy(MP+MAP_MP(l),MP+MAP_MP(l-1),l);

			int pos = LDSgetPos(MP+MAP_MP(l),tam);			
			MP[MAP_MP(l)+pos] = last[l-1];
	}
	return lmax;
}