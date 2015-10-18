for i in 17
do
	echo $i >> saida.txt
	./a.out $i 10240 >> saida.txt
done
