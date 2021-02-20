# Cancer Classification from single-cell RNA Sequencing Data using Dilated Convolutional Neural Networks

[The Cancer Genome Atlas (TCGA)](https://www.nature.com/articles/ng.2764) is a major effort to collect vast amounts of information on thousands of distinct tumor samples. This dataset is commonly referred to as the "pan-cancer" dataset. The dataset provides the community with a wide range of data on DNA alterations, gene expression, methylation status, protein abundances etc. These data prove to be useful in diagnosing many types of cancers. [Bailey et al.](https://pubmed.ncbi.nlm.nih.gov/29625053/) recently combined all 33 TCGA datasets to outline a pan-cancer map of which mutations can be drivers for the progression on cancer. Our project focuses on the said pan-cancer dataset. We suggest the use of modern deep learning tools to predict the cancer class from the single cell RNA Sequence data of the TCGA pan-cancer dataset. 


# RNA Sequencing

RNA-Seq is a technique that utilizes NGS (Next Generation Sequencing) techniques to identify the quantity and sequences of RNA in a sample. This technique analyzes the transcriptome of gene expression patterns encoded within RNA. 

# The TCGA Pan-Cancer Dataset

The TCGA pan-cancer dataset consists of gene expression for 20,530 genes obtained from 10,458 samples taken from cancerous tissue. The gene expression was measured using the IlluminaHiSeq sequencing device. The gene expression values are expressed as $'log2(x+1)'$ where $x$ is the normalized counts of each gene. The phenotype of each sample was classified into 33 categories including thyroid carcinoma, lung squamous cell carcinoma, breast invasive carcinoma and acute myeloid leukemia to name a few. The number of occurrences and the distribution of each of these classes present in the dataset is provided in the following Table.

