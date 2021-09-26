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
[[7]]

[0]: https://geometricdeeplearning.com/
[1]: https://www.cs.mcgill.ca/~wlh/grl_book/
[2]: https://distill.pub/2021/gnn-intro/
[3]: https://arxiv.org/abs/1403.6652
[4]: https://arxiv.org/abs/1607.00653
[5]: https://arxiv.org/abs/1503.03578
[6]: https://arxiv.org/abs/1803.04742

