## Efficient PreTrained Language Models

After the advent of BERT, many pretrained language models were introduced, and most of them opted for larger model sizes to achieve better performance. While large-scale pretrained models indeed ensure good performance, they come with the drawback of being challenging to use in typical computing environments. To address this issue, there has been a movement to build efficient pretrained models that can offer a certain level of performance.

There are broadly two ways to enhance model efficiency: one is by reducing the number of model parameters, and the other is by improving the attention mechanism. This project compares representative models of these two approaches with the baseline model, BERT, and assess the efficiency gains in real tasks. Models aimed at lightweighting include ALBERT, Distil BERT, and Mobile BERT, while those focused on enhancing attention mechanisms include Reformer, Longformer, and BigBird.

<br>


## Model Descs


<br>

> **LightWeight Focused Models**


* **ALBERT** <br>
  A Lite BERT
<br>
  
* **Distil BERT** <br>
Distilled BERT
<br>
  
* **Mobile BERT** <br>

<br><br>

> **Attention Focused Models**

* **Reformer** <br>

<br>

* **Longformer** <br>

<br>

* **BigBird**

<br><br>

## Model Specs

> **LightWeight Focused Models**

| Model | Params | Size | LightWeight Ratio (BERT Based) |
|:---:|:---:|:---:|:---:|
| BERT                      | &emsp; 109,482,240 &emsp; | &emsp; 417.649 MB &emsp; |  100%  |
| AlBERT                    |  11,683,584               |  44.577 MB               | &emsp; 10.67% &emsp; |
| Distil BERT               |  66,362,880               | 253.158 MB               | 60.62% |
| &emsp; Mobile BERT &emsp; |  24,581,888               |  93.776 MB               | 22.45% |

<br><br>

> **Attention Focused Models**

| Model | Params | Size | Attention Type |
|:---:|:---:|:---:|:---:|
| BERT                     | &emsp; 109,482,240 &emsp; | &emsp; 417.649 MB &emsp; | Full Attention |
| Reformer                 | 148,654,080               | 567.070 MB               | &emsp; Sparse Attention &emsp; |
| &emsp; Longformer &emsp; | 148,659,456               | 567.091 MB               | - |
| Big Bird                 | 127,468,800               | 486.317 MB               | - |

<br><br>

## Results

> **LightWeight Focused Models**

|  | BERT | AlBERT | Distil BERT | Mobile BERT |
|:---:|:---:|:---:|:---:|:---:|
| COLA Accuracy            | - | - | - | - |
| Training Speed per Batch | - | - | - | - |

<br><br>

> **Attention Focused Models**

|  | BERT | Reformer | Longformer | Big Bird |
|:---:|:---:|:---:|:---:|:---:|
| IMDB Accuracy            | - | - | - | - |
| Training Speed per Batch | - | - | - | - |

<br>

## How to Use
```
python3 run.py -mode ['lightweight', 'attention']
```
<br><br>

## References
* [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805)
* [**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**](https://arxiv.org/abs/1909.11942)
* [**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**](https://arxiv.org/abs/1910.01108)
* [**MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**](https://arxiv.org/abs/2004.02984)
* [**Reformer: The Efficient Transformer**](https://arxiv.org/abs/2001.04451)
* [**Longformer: The Long-Document Transformer**](https://arxiv.org/abs/2004.05150)
* [**Big Bird: Transformers for Longer Sequences**](https://arxiv.org/abs/2007.14062)

<br>
