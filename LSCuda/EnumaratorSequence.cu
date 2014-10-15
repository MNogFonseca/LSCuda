#include <stdio.h>
#include <stdlib.h>

void getSequence(int* dest, int n, int index){
	int i;
	//Cria o vetor primário de tamanho N
	for(i = 0; i < n; i++)
		dest[i] = i+1;

	//Percorre o vetor
	for(i = 0; i < n-1; i++){
		//Calcula quantas alterações são possiveis sem alterar o primeiro elemento atual
		int fat = fatorial(n-i-1);
		//Calcula quantas vezes foi possível trocar essa posição
		int num_movimentos = index/fat;
		if(num_movimentos > 0){
			shiftElement(dest, num_movimentos);
			//Diminui a quantidade ja calcula do indice
			index -= num_movimentos*fat;
		}
		dest++;
	}
}

int getIndex(int* vet, int n){
	int i;
	int index = 0;
	for(i = 0; i < n-1; i++){
		index += (vet[i]-(i+1))*fatorial(n-i-1);
	}
	return index;
}