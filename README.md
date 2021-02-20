# Cancer Classification from single-cell RNA Sequencing Data using Dilated Convolutional Neural Networks

[The Cancer Genome Atlas (TCGA)](https://www.nature.com/articles/ng.2764) is a major effort to collect vast amounts of information on thousands of distinct tumor samples. This dataset is commonly referred to as the "pan-cancer" dataset. The dataset provides the community with a wide range of data on DNA alterations, gene expression, methylation status, protein abundances etc. These data prove to be useful in diagnosing many types of cancers. [Bailey et al.](https://pubmed.ncbi.nlm.nih.gov/29625053/) recently combined all 33 TCGA datasets to outline a pan-cancer map of which mutations can be drivers for the progression on cancer. Our project focuses on the said pan-cancer dataset. We suggest the use of modern deep learning tools to predict the cancer class from the single cell RNA Sequence data of the TCGA pan-cancer dataset. 


# RNA Sequencing

RNA-Seq is a technique that utilizes NGS (Next Generation Sequencing) techniques to identify the quantity and sequences of RNA in a sample. This technique analyzes the transcriptome of gene expression patterns encoded within RNA. 

# The TCGA Pan-Cancer Dataset

The TCGA pan-cancer dataset consists of gene expression for 20,530 genes obtained from 10,458 samples taken from cancerous tissue. The gene expression was measured using the IlluminaHiSeq sequencing device. The gene expression values are expressed as log2(x+1) where x is the normalized counts of each gene. The phenotype of each sample was classified into 33 categories including thyroid carcinoma, lung squamous cell carcinoma, breast invasive carcinoma and acute myeloid leukemia to name a few. The number of occurrences and the distribution of each of these classes present in the dataset is provided in the following Table.

<p align="center">
    <img src="https://raw.githubusercontent.com/suhailnajeeb/tcga-cancer-predict/master/images/table.jpg">
</p>

In this work, the dataset was split into stratified 5-folds such that each fold retained the same proportion of the classes as the full dataset. The models presented in this paper were then trained on all folds except one, on which it was evaluated. This was repeated until all the folds served as the evaluation set once.

# Data Visualization

To get a better understanding of the RNASequence distribution, we used t-SNE (t-distributed Stochastic Neighbour Embedding) for dimensionality reduction. Each RNASeq sample was reduced to two components from 20,530 RNASeq features.

<p align="center">
    <img src="https://raw.githubusercontent.com/suhailnajeeb/tcga-cancer-predict/master/images/TSNE.png">
</p>

# 1D Convolutional Neural Network Architecture

We first developed a 1D Convolutional Neural Network to fit the RNASeq data. Since each sample of the data is one-dimensional in nature, we employed 1D convolutions to the network. We tried different architectures for 1D CNN. First, a basic conv x 2 and pool with fully connected layers was used. Although this model performed well, it showed overfitting. Adding a second pair of conv-pool improved results, however, the model size remained quite heavy. Not all genes have a same level of importance while performing the prediction task. Therefore, it is necessary to permute and optimize the locations of the genes in the array. A fully connected layer at the input serves this purpose and at the same time reduces the model size by a significant margin. The architecture of our final 1D CNN model is shown below. A flattening operation precedes another fullly connected layer before the final softmax layer at the end of the network. The softmax layer produces probabilities of the respective class. The prediction is decided by the class having the highest probability.

<p align="center">
    <img src="https://raw.githubusercontent.com/suhailnajeeb/tcga-cancer-predict/master/images/1dcnn_model.jpg">
</p>

The model was trained using the Adam optimizer for 20 epochs, using dropout and learning rate reduction when plateaus were reached in the validation loss. The final model was trained using stratified k-fold cross validation. The accuracy of the model was found 94.9% over the 5 folds. 

