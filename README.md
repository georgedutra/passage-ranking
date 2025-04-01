# passage-ranking
Repositório de trabalho com o dataset passage ranking msmarco

# Sumário
- [Criação de sub-set](#criação-de-sub-set)
  - [Escolha de um conjunto de dados válido](#escolha-de-um-conjunto-de-dados-válido)
    - [Proporção do tipo de pergunta](#proporção-do-tipo-de-pergunta)
    - [Scores dos documentos](#scores-dos-documentos)
  - [Como criar?](#como-criar)

# Criação de sub-set
Dado o conjunto de dados MS-MARCO é necessário criar um sub-set dos dados originais de forma aleatória para gerar modelos estatísticos em cima desses dados.

## Escolha de um conjunto de dados válido
Para criar um conjunto de dados válido em menor escala e assim treinar modelos para tarefas específicas, é necessário manter o mesmo viés dos dados originais.

### Proporção do tipo de pergunta
O conjunto de dados original apresenta um paper relacionado que descreve quais os tipos de perguntas existem e suas porcentagens correspondentes. Como o intuito de manter esse viés, o sub-set deve selecionar dados aleatórios mas que ainda mantenha a mesma proporção de tipos de perguntas.

No dataset original existem:
| Tipo de Pergunta | Porcentagem |
|-----------------|------------|
| Yes/No         | 7,46%      |
| What          | 34,96%     |
| How           | 16,8%      |
| Where         | 3,46%      |
| When          | 2,71%      |
| Why           | 1,67%      |
| Who           | 3,33%      |
| Which         | 1,79%      |
| Other         | 27,83%     |

### Scores dos documentos
Uma das tarefas específicas do paper é saber se é possível resolver uma pergunta baseado nos documentos exibidos. Portanto para garantir que essa informação não seja perdida no dataset, ao selecionar os documentos correspondentes a uma query, é necessário manter documentos de alta e baixa relevância. Para essa tarefa foi utilizado o score já calculado pelos documentos para o dataset.

## Como criar?
Chame a função create_sub_dataset(n) que será retornado ... com n perguntas aleatórias com o processo descrito e seus documentos correspondentes.


## Abordagem "Bert-only"
O Bert é um modelo que extrai conteúdo semântico de sentenças. Ou seja, ele produz um embedding enriquecido pela self-attention entre os tokens e é esse embedding que é passado para os modelos seguintes que fazem as tarefas downstream. No caso do bert-base-uncased usado, o embedding de cada documento é um vetor de tamanho 768. Nós carregamos cada um dos documentos e criamos uma matrix n_docs x 768 que descreve o corpus de documentos.

A nossa ideia é que queries são bem respondidas por documentos que estão em campos semânticos similares. Ou seja, uma query é bem respondida por um documento com um embedding similar. Essa similaridade é avaliada pelo produto interno entre embeddings.

Assim, para sugerir documentos que respondem uma query, o procedimento é:
- carregar a matrix $D$ de embeddings de documentos
- passar a query pelo bert para obter o embedding $q$
- calcular $s = Dq$
- os documentos que melhor respondem a query são os k argmax de $s$

Importante: criamos um dicionário que mapeia esses índices para o id original do documento.

A maneira mais eficiente de avaliar o modelo seria criar uma matrix de embeddings das queries $Q$ e calcular $DQ$, mas, para simular um uso real (em que uma aplicação recebe diferente queries em sequência), não agregamos cada um dos vetores de embeddings de queries em uma matriz, mas calculamos os embeddings em tempo de execução e fazemos $Dq$.

## Métricas
O que chamamos de precisão@x é se pelo menos umas das x primeiras sugestões é relevante

Tempo médio para recuperar um documento para uma query: 0.013438280638273771 segundos
Precisão do modelo no teste: 0.866113316492241 %
Precisão@5 do modelo no teste: 3.067484662576687 %
Precisão@10 do modelo no teste: 5.44929628293035 %
MRR@5: 0.016426079634307714
MRR@10: 0.019555859153477344