---
title: 'Лекция 8'
date: 2021-09-24
permalink: /lectures/lecture8
toc: true
sidebar:
  nav: "lectures"
tags:
  - embeddings
  - graph ml
  - gnn
  - message passing
  - graph neural network
  - knowledge graph embeddings
  - gcn
---

## Graph Neural Networks and KGs

| Материалы |  Ссылка  |
 ------------- | ------------- |
 Видео  | [YouTube](https://youtu.be/_aX-YSIIn0k) | 
 Слайды  | [pdf](/kgcourse2021/assets/slides/Lecture8.pdf) |
 Конспект |  [здесь](https://migalkin.github.io/kgcourse2021/lectures/lecture8)  |
 Домашнее задание | [link](#домашнее-задание) |

## Видео

<iframe width="560" height="315" src="https://www.youtube.com/embed/_aX-YSIIn0k" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Graph Encoders

В предыдущей лекции мы рассматривали простые (shallow) decoder-only модели для векторных представлений KG и их использовании только в задаче предсказания связей (link prediction). Область машинного обучения на графах (Graph ML), однако, куда шире и позволяет решать множество других задач, в т.ч. классификацию вершин (node classification), классификацию графов (graph classification), регрессии вершин и графов (node regression, graph regression). Одним из мощных инструментов решения этих задач, который пригодится и в исследовании KGs являются графовые нейросети (Graph Neural Networks, GNNs), которые могут служить универсальным графовым энкодером для решения downstream задач.

![](/kgcourse2021/assets/images/l8/l8_p1.png)

**Message Passing** (передача сообщений) - один из подходов для построение и анализа GNNs. Большое количество GNN архитектур можно интерпретировать в контексте message passing, поэтому сперва стоит рассмотреть основы этой парадигмы. 
Содержимое этой лекции основано на нескольких книгах, **Geometric Deep Learning protobook** [[0]], **Graph Representation Learning Book** [[1]], где можно найти более подробные теоретические обоснования, выкладки и объяснения. Еще стоит отметить публикации в журнале Distill.pub [[2]].

В этой лекции мы рассмотрим концепцию message passing, как она работает на классических (не multi-relational) графах, основные семейства архитектур, а затем их усовершенствования для работы с KGs, которые отличаются (1) наличием типов связей; (2) частым отсутствием признаков вершин, из-за чего нужно обучать эмбеддинги вершин.

### Inputs & Outputs

Прежде чем рассматривать конкретные имплементации, стоит договориться о формате ввода-вывода от ожидаемых архитектур:

![](/kgcourse2021/assets/images/l8/l8_p2.png)

Как правило, на вход графовым энкодерам в message passing подаются:
* Матрица смежности (adjacency matrix) $A$ - часто для экономии памяти для графа из $N$ вершин вместо квадратной $N\times N$ матрицы используются разреженные матрицы (sparse matrix) или листы ребер (edge list).

![](/kgcourse2021/assets/images/l8/l8_p3.png)

* Признаки вершин (node features) $X\in \mathbf{R}^{\|N\|\times d}$ - которые могут быть заданными или обучаемыми эмбеддингами.
* (Опционально) Признаки ребер (edge features)  $E\in \mathbf{R}^{\|E\|\times d}$
* (Опционально) Типы ребер (relation types) $R\in \mathbf{R}^{\|R\|\times d}$

На выходе процедуры message passing - обозначим ее функцией $f(.)$ - получаются обновленные представления всех признаков, поданных на вход:
* $f(X) \rightarrow X'$
* $f(E) \rightarrow E'$
* $f(R) \rightarrow R'$

Обновленные представления из графового энкодера затем можно использовать в конкретной задаче, например, классификации вершин или предсказании связей, подключив подходящий декодер.

Message passing - не единственный способ строить графовые энкодеры и векторные представления, например, существуют и успешно применяются unsupervised методы (DeepWalk [[3]], node2vec [[4]], LINE [[5]], VERSE [[6]]). Тем не менее, из-за наличия многочисленных типов связей в KGs, message passing архитектуры получили более широкое применение в контексте работы с графами знаний.

### Message Passing: Aggregate & Update

Идея message passing проста - вершины графа посылают _сообщения_, и новое представление каждой вершины получается как функция $\phi$ от:
1. Предыдущего представления вершины
2. Сообщений соседей
   
Эту функцию можно записать как:

\\[ \mathbf{h}\_{u} = \phi (\mathbf{x}\_{u}, \mathbf{X}\_{\mathcal{N}(u)}) \\]


где $\mathbf{x}_u$ - предыдущее представление вершины $u$, $\mathbf{X}_{\mathcal{N}(u)}$ - представления соседей $u$, где $\mathcal{N}(u)$ обозначает сообщество вершины $u$. Другими словами, message passing выполняет процесс итеративной агрегации соседей (neighborhood aggregation).

Представление соседей получается путем как функция $\psi$ от представлений соседей:

\\[ \mathbf{X}_{\mathcal{N}(u)} = \psi (\mathbf{x}_{n1}, \dots , \mathbf{x}_{nk}) \\]

![](/kgcourse2021/assets/images/l8/l8_p31.png)

Для графа на рисунке для вершины $b$ представление соседей строится от представлений вершин $a$, $c$, $d$. Часто, для простоты в функцию $\psi$ добавляют и предыдущее представление рассматриваемой вершины $v$:

\\[ \mathbf{X}_{\mathcal{N}(b)} = \psi (\mathbf{x}_a,  \mathbf{x}_b, \mathbf{x}_c, \mathbf{x}_d ) \\]

Концептуально, message passing состоит из двух шагов:
1. Построение сообщения через агрегацию соседей (Aggregate)
2. Обновление представления вершины (Update)

![](/kgcourse2021/assets/images/l8/l8_p4.png)

**Шаг Aggregate**

На первом шаге для вершины $u$ строится сообщение $\mathbf{m}_{\mathcal{N}(u)}$:

\\[ \mathbf{m}_{\mathcal{N}(u)} = \text{AGGREGATE}( \\{ \mathbf{h}_v, \forall v \in \mathcal{N}(u) \\} ) \\]

В графах нет простого понятия "местоположения" вершины, то есть мы не можем сказать, что вершина $u$ находится "справа" или "сверху" от вершины $v$. У каждой вершины есть сообщество соседей, которое мы можем в общем случае перечислять в любом порядке. Поэтому функция агрегации должна быть инвариантна к перестановкам (**permutation invariance**) - то есть результат агрегации не зависит от порядка ее применения к вершинам-соседям.

Мы будем записывать permutation invariant функции как $\bigoplus$ :

\\[ \mathbf{m}_{\mathcal{N}(u)} = \bigoplus_{v \in \mathcal{N}(u)} \psi ( \mathbf{x}_v ) \\]

Простые 4 функции, инвариантные к перестановке аргументов:
* суммирование $\sum$
* усреднение $avg()$
* взятие минимального $min()$
* взятие максимального $max()$

Например, часто используемое агрегирование через сумму представлений соседей и умножение с обучаемой весовой матрицей $\mathbf{W}_{\text{neigh}}$ будет записываться как:

\\[ \mathbf{m}_{\mathcal{N}(u)} = \mathbf{W}_{\text{neigh}}\sum_{v \in \mathcal{N}(u)} \mathbf{x}_v \\]

В зависимости от задачи, выбор инвариантной функции может заметно влиять на результат [[7]] - например, если в задаче определения изомрофизма двух графов с разным количеством вершин у всех вершин одинаковые признаки, то функции $avg(), min() , max()$ вернут одинаковые значения и только $sum()$ вернет уникальные значения. 
Еще можно использовать все инвариантные функции сразу и составлять сообщение как линейную комбинацию этих функций (метод Principal Neighborhood Aggregation, PNA) [[8]].

**Шаг Update** 

Новое представление вершины $u$ получается в результате функции $\text{UPDATE}$ от предыдущего представления этой вершины $\mathbf{h_u}$ и сообщения $\mathbf{m}$, полученного на шаге AGGREGATE:

\\[ \mathbf{h}_{u}^{(k+1)} = \text{UPDATE}( \mathbf{h}_u^k, \mathbf{m}_{\mathcal{N}(u)}^k ) \\]

Или с использованием нотации агрегирования:

\\[ \mathbf{h}_{u}^{(k+1)} = \phi( \mathbf{h}_u^k, \bigoplus_{v \in \mathcal{N}(u)} \mathbf{h}_v ) \\]

В простейшем виде функция UPDATE может складывать преобразованные представления и пропускать результат через некоторую нелинейную функцию $\sigma$ (sigmoid, tanh, ReLU, и т.д.):

\\[ \mathbf{h}_{u}^{(k+1)} = \sigma( \mathbf{W}_{\text{self}}\mathbf{h}_u^k + \mathbf{W}_{\text{neigh}} \mathbf{m}_{\mathcal{N}(u)}^k  ) \\]

где $\mathbf{W}_{\text{self}}$ - обучаемая весовая матрица предыдущего представления, $\mathbf{W}_{\text{neigh}}$ - весовая матрица агрегации соседей из шага AGGREGATE.

В целом, задача функции UPDATE - скомбинировать имеющиеся векторы в новое представление вершины, поэтому способов такой комбинации может существовать довольно много и иметь разную сложность (например, использовать реккурентные модули (GRU или LSTM) [[9]]).

### Глубина Message Passing сетей



### Матричная запись

## Message Passing Архитектуры

![](/kgcourse2021/assets/images/l8/l8_p5.png)

### Graph Convolutional Nets (GCN)

### Graph Attention Nets (GAT)

### Message Passing Neural Nets (MPNN)

## Relational GCNs (R-GCN)

## Compositional GCNs (CompGCN)

## Inductive Learning

### Out-of-Sample Learning

### Textual Features

### Structural Features 



## Библиотеки и репозитории

Популярные библиотеки для работы с GNNs:
* [PyG](https://github.com/pyg-team/pytorch_geometric) (PyTorch)
* [DGL](https://github.com/dmlc/dgl) (PyTorch, MXNet, TensorFlow)
* [Jraph](https://github.com/deepmind/jraph) (Jax)

## Домашнее задание


## Использованные материалы и ссылки:

[[0]] Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. 2021   
[[1]] William L. Hamilton. Graph Representation Learning. Morgan and Claypool. 2020   
[[2]] Sanchez-Lengeling, et al., "A Gentle Introduction to Graph Neural Networks", Distill, 2021.   
[[3]] Bryan Perozzi, Rami Al-Rfou, Steven Skiena. DeepWalk: Online Learning of Social Representations. KDD 2014.   
[[4]] Aditya Grover and Jure Leskovec. node2vec: Scalable Feature Learning for Networks. KDD 2016   
[[5]] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei. LINE: Large-scale Information Network Embedding. WWW 2015   
[[6]] Anton Tsitsulin, Davide Mottin, Panagiotis Karras, Emmanuel Müller. VERSE: Versatile Graph Embeddings from Similarity Measures. WWW 2018   
[[7]] Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. How Powerful are Graph Neural Networks?. ICLR 2019   
[[8]] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković. Principal Neighbourhood Aggregation for Graph Nets. NeurIPS 2020   
[[9]] Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel. Gated Graph Sequence Neural Networks. ICLR 2016   
[[10]]

[0]: https://geometricdeeplearning.com/
[1]: https://www.cs.mcgill.ca/~wlh/grl_book/
[2]: https://distill.pub/2021/gnn-intro/
[3]: https://arxiv.org/abs/1403.6652
[4]: https://arxiv.org/abs/1607.00653
[5]: https://arxiv.org/abs/1503.03578
[6]: https://arxiv.org/abs/1803.04742
[7]: https://openreview.net/forum?id=ryGs6iA5Km
[8]: https://arxiv.org/abs/2004.05718
[9]: https://arxiv.org/abs/1511.05493
