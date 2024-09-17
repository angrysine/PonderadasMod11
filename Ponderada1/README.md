# Ponderada 1

## Descrição

Nesse ponderada foi desenvolvida uma ula de 8 bits com a soma,subtração, multiplicação, comparação e deslocamento a esquerda. A alu foi juntada a dois displays de 7 segmentos para mostrar o resultado da operação.
A Alu recebe 3 parâmetros A, B e Op, onde A e B são os valores que serão operados e Op é a função que será realizada. A função é selecionada através de um switch de 3 bits. O resultado da comparação é mostrado no em 3 saidas adicionais que indicam se A é maior, menor ou igual a B. O circuito que contem o sistema final se chama DisplayUla.dig .

## Operações

| Operação       | Código        |
|----------------|---------------|
| 0              | add           |
| 1              | compare       |
| 2              | subtraction   |
| 3              | shift         |
| 4              | multiplication|

## Vídeo

[Link do vídeo](https://youtu.be/vuOrVeMqA6o)


## Desafio Extra\

No arquivo FDX.dig foi feito uma implementação parcial do ciclo fdx, ataualmente o sistema é capaz de guardar a próxima instrução no registrador de instrução, é capaz de separar a operação e o endereço do operando, porém ainda não é capaz de realizar a operação. O que falta para o sistema funcionar é transformar o ciclo atual que é de tick de clock em um de 3 ticks, onde no primeiro tick é feita a busca da instrução, no segundo é feita a decodificação e no terceiro é feita a execução.