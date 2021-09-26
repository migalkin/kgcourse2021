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
Содержимое этой лекции основано на нескольких книгах, **Geometric Deep Learning protobook** [[0]], **Graph Representation Learning Book** [[1]], где можно найти более подробные теоретические обоснования, выкладки и объяснения.

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

На выходе процедуры message passing (обозначим ее функцией $f(.)$) получаются обновленные представления всех признаков, поданных на вход:
* $f(X) \rightarrow X'$
* $f(E) \rightarrow E'$
* $f(R) \rightarrow R'$

### Message Passing: Aggregate & Update



### Graph Convolutional Nets (GCN)

### Graph Attention Nets (GAT)

### Message Passing Neural Nets (MPNN)

## Relational GCNs (R-GCN)

## Compositional GCNs (CompGCN)

## Inductive Learning

### Out-of-Sample Learning

### Textual Features

### Structural Features 



![](/kgcourse2021/assets/images/l7/l7_p1.png)


## Библиотеки и репозитории

В практической части мы будем использовать библиотеку [PyKEEN](https://github.com/pykeen/pykeen/), где содержатся большинство описанных моделей, способов тренировки, оптимизации, и других компонентов.

Популярные библиотеки для работы с KG embedding:
* [PyKEEN](https://github.com/pykeen/pykeen/) (PyTorch)
* [LibKGE](https://github.com/uma-pi1/kge) (PyTorch)
* [OpenKE](https://github.com/thunlp/OpenKE) (PyTorch / TensorFlow)
* [AmpliGraph](https://github.com/Accenture/AmpliGraph) (TensorFlow)
* [GraphVite](https://github.com/DeepGraphLearning/graphvite) (Python / C++)
* [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) (PyTorch)
* [DGL-KE](https://github.com/awslabs/dgl-ke) (PyTorch / TensorFlow / MXNet)
* [pykg2vec](https://github.com/Sujit-O/pykg2vec) (PyTorch) 

## Домашнее задание

1. Выведите формулу ComplEx $\text{Re}\langle h, r, \bar{t} \rangle $ через конкретные действительные и мнимые части сущностей и предикатов как результат умножения трех комплексных чисел. Пусть каждое комплексное число $x$ состоит из $x_{re} + i \cdot x_{im}$.
2. Выведите формулу Hadamard product между комплексными сущностью и предикатом RotatE $ h \circ r $. Представьте $r$ как через матрицу вращения, где $\text{cos}(r)$ можно положить за $r_{re}$ и  $\text{sin}(r)$ как $r_{im}$. Эта матрица вращает $h$ - вектор-столбец из действительной и мнимой частей.
3. [Colab Notebook](https://colab.research.google.com/drive/1m8K1gFZqv2tDKU8vKfIoi1qOHbuycC2z) с набором заданий и примеров работы с KG embedding алгоритмами с библиотекой PyKEEN.

## Использованные материалы и ссылки:

[[0]] Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. 2021   
[[1]] William L. Hamilton. Graph Representation Learning. Morgan and Claypool. 2020   
[[2]]  

[0]: https://geometricdeeplearning.com/
[1]: https://www.cs.mcgill.ca/~wlh/grl_book/
[0]: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7358050&casa_token=e0MzaF2_ZnkAAAAA:hMhTmlqixbqnhjvIC2VHnb3qhFAapnwY1wXrsXt6L6BilJJwcWBgwaMh3NLu13WF60Hw7e4&tag=1
[1]: https://distill.pub/2021/gnn-intro/

