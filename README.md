# Multi-label-Protein-Function-Prediction
## Datasets used:
### https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data (Dataset containing protein sequences taken from a competition on kaggle)
### https://www.kaggle.com/datasets/sergeifironov/t5embeds (Dataset containing vector embedding generated using T5 transformer model)
### https://www.kaggle.com/datasets/yashvashistha1/labels (Labels extracted from the protein sequences by us to save time in running the code for generation of labels everytime).

## Files description:
`multi-label-protein-function-prediction.ipynb`: Notebook to train the model and run inference

`sequence_from_fasta.ipynb`: Helper notebook to extract protein sequence from a fasta file

`labels.csv`: Labels extracted from the dataset

`example.fasta`: Example fasta file

`gradio.py`: Used in hugging face to create an interface

`requirements.txt`: Libraries and packages needed 

## How to test the code: 
### Download the multi-label-protein-function-prediction.ipynb notebook open it in kaggle(prefered) or google colab(don't forget to change the dataset paths to avoid errors), load the datasets and run the cells you should see the model training.
### Alternatively just click the link: https://www.kaggle.com/code/yashvashistha1/ci-project-v3
### Then click on copy and edit and click run all (remember to select gpu accelaration)
### Model deployment can be seen on the following link: https://huggingface.co/spaces/Dinesh1102/Protein_Function_Prediction (Inference could be slower because it is running on CPU resources not GPU).
### Files used for deploying are: https://huggingface.co/spaces/Dinesh1102/Protein_Function_Prediction/tree/main
