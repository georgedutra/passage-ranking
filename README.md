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
