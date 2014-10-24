#include <stdio.h>
#include <stdlib.h>

unsigned long fatorial(int n){
	unsigned long result = 1;
	for(int i = n; i > 1; i--){
		result *= i;
	}
	return result;
}

//Coloca o elemento da variável pos na primeira posição, e da um shift para a direia nos outros
void shiftElement(char* dest, int pos){
	char temp = dest[pos];
	for(int i = 0; i < pos; i++){
		dest[pos-i] = dest[pos-i-1];
	}
	dest[0] = temp;
}

void getSequence(char* dest, int n, unsigned long index){
	//Cria o vetor primário de tamanho N
	for(int i = 0; i < n; i++)
		dest[i] = i+1;

	//Percorre o vetor
	for(int i = 0; i < n-1; i++){
		//Calcula quantas alterações são possiveis sem alterar o primeiro elemento atual
		unsigned long fat = fatorial(n-i-1);
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

int qtdMenores(char* vet, int num, int n){
	int qtd = 0;
	int i;
	for(i = 0; i < n; i++){
		if(vet[i] < num)
			qtd++;
	}
	return qtd;
}

unsigned long getIndex(int* vet, int n){
	int i;
	unsigned long index = 0;

	for(i = 0; i < n-1; i++){
		index += qtdMenores(vet+i+1, vet[i], n-i-1)*fatorial(n-i-1);
	}
	return index;
}