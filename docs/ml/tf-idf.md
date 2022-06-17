# TF-IDF

Term Frequency - Inverse Document Frequency
- Used in Information Retrieval to determine the importance of a **word** within a **document**
- Used for standardizing the importance of a word between two documents of different total word count.
- **Term Frequency**
    - $\text{tf}_{(t,d)} = \frac{n_{t,d}}{\sum^{T}_{k=1} n_{k,d}}$
    - $t$ : term
    - $d$ : document
    - $n_{t,d}$ : term frequency/count
    - $\sum^{T}_{k=1} n_{k,d}$ : sum of words in document *d*
- **Inverse Document Frequency**
    - $\text{idf}_{(t,D)} = log\bigg(\frac{D}{d_t}\bigg)$
    - $d_t$ : number of documents that have term *t*
    - $D$ : total number of documents
- **TF-IDF**
    - $\text{score}_{(t,d,D)} = tf_{(t,d)} \times idf_{(t,D)}$
    - If *t* has a high frequency in *d*, the value of $tf_{(t,d)}$ would be large.
    - If *t* has a lower frequency in other documents, the value of $idf_{(t,D)}$ would be large

In TF-IDF, each term represents a vector; and the number of vectors determine the dimension of the model. Thus, removing unnecesary terms would reduce the complexity of our model.
