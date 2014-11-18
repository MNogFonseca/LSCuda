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

void getSequenceLexicographically(char* dest, int n, unsigned long index){
	//Cria o vetor primário de tamanho N
	for(int i = 0; i < n; i++)
		dest[i] = i+4; //Começar do numero 4

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

void getSequence(char* dest, int n, unsigned long index){
	int numDeslocamentos2e3 = index/fatorial(n-3);
	int indexResto = index%fatorial(n-3);
	
	int pos_num2 = 1;
	int pos_num3;
	int i;
	for(i = 0; numDeslocamentos2e3; i++){
		if(numDeslocamentos2e3 > (n-2-i)){
			pos_num2++;
			numDeslocamentos2e3 -= (n-2-i);
		}
		else{
			pos_num3 = pos_num2 + 1 + numDeslocamentos2e3;
		}
	}
	getSequenceLexicographically(dest+3, n-3, indexResto);
	dest[0] = 1;

	for(i = 1; i < pos_num2; i++){
		dest[i] = dest[i+2];
	}
	dest[pos_num2] = 2;

	for(i = pos_num2+1; i < pos_num3; i++){
		dest[i] = dest[i+1];
	}
	dest[pos_num3] = 3;
	shiftEsquerda(dest, n, )
}

/*0 - 1 2 3 4 5 6 7
1 - 1 2 4 3 5 6 7
2 - 1 2 4 5 3 6 7
3 - 1 2 4 5 6 3 7
4 - 1 2 7 4 5 6 3
5 - 1 4 2 3 5 6 7
6 - 1 4 2 5 3 6 7
7 - 1 4 2 5 6 3 7
8 - 1 4 2 5 6 7 3
9 - 1 4 5 2 3 6 7
A - 1 4 5 2 6 3 7
B - 1 4 5 2 6 7 3
C - 1 4 5 6 2 3 7
D - 1 4 5 6 2 7 3
E - 1 4 5 6 7 2 3
*/


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

/*
int main(){
	char[7] vetor;
	int i = 0;
	for(; i < 10; i++){
		getSequence(vetor, i);
		int j = 0;
		for(; j < 7; j++){
			printf("%c - ",vetor[j]);
		}
		printf("\n");
	}
}*/