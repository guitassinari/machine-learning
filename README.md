# random-forest

### Python

Deve ser utilizada a versão Python 3.7 ou superior

### Bibliotecas

#### Pandas
https://pandas.pydata.org/

Instalar usando

```
pip3 install pandas
```

#### Sklearn
https://scikit-learn.org/stable/

Instalar usando

```
pip3 install sklearn
```

#### Matplotlib
https://matplotlib.org/

Instalar usando

```
pip3 install matplotlib
```

### Estrutura

#### Passo a passo

```
Precisamos validar os passos a seguir com os professores
e garantir que estão corretos
```

1. Ler arquivo de dataset
2. Dividir o dataset em grupos e agrupá-los de maneira estratificada em:
    - dataset de treino
    - dataset de validação
    - dataset de teste
3. Para cada configuração de hiperparâmetros,
criar uma Floresta Aleatória passando o dataset de treinamento
4. Cada Floresta deve, a partir do dataset de treinamento, através do método
de Bootstrap com reposição (Bagging), gerar outros sets de treinamento, que devem ser
usados para a indução de suas árvores de decisão
5. Utilizando o dataset de validação, avaliar o desempenho de cada Floresta
na predição das classes do mesmo
6. Compilar os desempenhos das florestas e selecionar a configuração de hiperparâmetros
com melhor desempenho
7. Re-treinar o modelo escolhido com o set de treinamento + validação

8. Validar o modelo escolhido com  o dataset de teste

( Repetir os passos 2  ~ 5 até que todos os agrupamentos do dataset original
tenham sido utilizados para treinamento, validação e teste ) `É isso mesmo?
`


```
Duvidas:

1. Quais são as métricas que devemos utilizar para avaliar desempenho?
Devemos diferenciar entre classificação e regressão? Que média e desvio padrão
são esses comentados na especificação?
2. Como compilaremos as métricas, dado que sabemos quais métricas utilizarmos
(duvida 1)
3. Para a Validação Cruzada Estratificada, devemos primeiro separar em
treino e teste para, depois, dividirmos treino em grupos utilizaros para treino
e validação? Temos que repetir os passos de cross validation para cada configuração
de hiperparâmetros? Se não, como deve funcionar a separação e a utilização dos sets
de treinamento, validação e teste?
```


Descrição do AUDIÂO kkkk

Vamos receber um conjunto de treinamento Dataset de treinamento que vamos passar para a floresta, e aí essa floresta que é um conjunto de árvores de decisão, uma diferente da outra, que vão ser treinadas a partir de variações nesse dataSet que a gente pegou. Por exemplo: Eu tenho um dataSet com exemplos de 1 a 10, então eu passo para a árvore 1 o conjunto com o exemplo 2 repetido (1,2,3,4,5,6,7,8,9,2), para a árvore 2 com o exemplo 7 repedito (1,2,3,4,7,6,7,8,9,10)... alguma coisa assim... Então eu treino varias árvores diferentes e aí quando nós fomos tomar decisões a partir de um novo exemplo, a gente pega todas as decisões dessa nossa floresta de árvores e através de votação majoritária, ou seja, da árvore que nós der o valor maior, ou seja a decisão maior que for tomada pela maioria das árvores a gente seleciona.

Como cada árvore vai ser treinada?
	Reposta: Ela recebe um DataSet de treinamento, e a gente tem que fazer toda a indução da árvore, construção da árvore de decisão, e aí é que entra o VALOR DE AMOSTRAGEM. Porque assim como faziamos no inicio, nas primeira aulas que vimos árvores de decisão?
	Reposta: Tu tinha um algorimo que de acordo com o ganho de informação para decidir o nodo RAIZ, e aí depois no próximo nível da árvore tu não considerava mais aquele atributo, ou seja, tu não considera mais o atributo que tu usou no nodo RAIZ, considerava (N-1 atributos), mo próximo nível da árvore (N-2 atributos), só que o que o algorimo de floresta aleatória faz?
	Resposta: Ao invés de fazer isso mencionado acima tu escolhe o atributo da raiz, e em todos os outros níveis, tu não vai considerar n-1 atributos, tu vai considerar a RAIZ QUADRADA de N atributos.

Exemplo: Digamos que tu tenha esses atributos de uma pessoa:

{Idade, peso , altura , tamanho do pe }

No nó raiz da primeira árvore tu escolheu (altura)...

Nodo do primeiro nível = ALtura.


Quando tu for decidir o nodo do segundo nível tu vai fazer RAIZ QUADRADA do número de atributos que tu tem. Então Raiz(4)= 2, nesse caso tu vai escolher dois atributos dentre esse conjunto de quatro atributos que tu tem, para escolher o nó raiz do próximo nível

tinha 4 espaços {  ,  ,  ,  }

Fez RaizQuadrada(4)= 2 <------

Ficou 2 espaços {  ,  }

, e mesmo que tu tenha escolhido o primeiro atributo "altura" no nível anterio, ele pode ser repedito novemente nesse subconjunto da tua segunda escolha que terá dois atributos, e dentre desses dois atributos tu vai decidir qual é o que vai ser utilizado no segundo nó como raiz. Aí faz esse algoritmo para todas as árvores, e de acordo com o professor, a partir disso tu tem dois pontos de randomicidade no algoritmo de floresta, um é no dataSet que é passado para a árvore com o algoritmo de Boostrap onde fazemos as modificações no dataSet original e a outra é essa de pode repetir atributos nos níveis de árvore e pegando apenas um pequeno conjunto deles a cada nó, que no caso pode ser a RaizQuadrada(N) ou um número que a gente decidir. A descrição do trabalho que a literatura propõe sempre a raiz quadrada do número de atributos mas pode ser o que a gente quiser. E aí a partir disso tu treina a árvore, faz a montagem da árvore e é GOLLLL. No fim de tudo isso pronto nós temos que, através de CrossValidadtion, poder validar e comparar a performance do nosso algoritmo com N árvores(dez árvores, quinze árvores...) e com número de atributos diferentes se assim quisermos.

primeira parte: (essa parte é bem grande) Treinamento de uma arvore de decisão, envolve calculo de entropia, envolve esse algoritmo de amostragem.

Segunda Parte: Dá floresta., envolve o algoritmo de boostraping, para definir o subconjunto de treinamento para cada árvore.

Terceira parte: Validação(fazer o voto majoritário de todas as decisões da árvores, para decidir qual é decisão final da nossa floresta) e comparação de performance.
