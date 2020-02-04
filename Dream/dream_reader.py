import pandas as pd
import numpy as np
from sklearn import preprocessing
class DreamData:
    def __init__(self, path_to_data, dream_nbr = 1, data_size = 100):
        print(path_to_data + '/Data/DREAM4_InSilico_Size' + str(data_size) + '/insilico_size' +
                         str(data_size) + '_' + str(dream_nbr) +'/insilico_size' + str(data_size) + "_" +
                         str(dream_nbr) +'_multifactorial.tsv')
        df = pd.read_csv(path_to_data + '/Data/DREAM4_InSilico_Size' + str(data_size) + '/insilico_size' +
                         str(data_size) + '_' + str(dream_nbr) +'/insilico_size' + str(data_size) + "_" +
                         str(dream_nbr) +'_multifactorial.tsv', delimiter = '\t')
        adjacence_df = pd.read_csv(path_to_data + '/Gold_standards/Size ' + str(data_size) +
                                   '/DREAM4_GoldStandard_InSilico_Size' + str(data_size) + '_' + str(dream_nbr) +
                                   '.tsv', delimiter = '\t', header = None)
        datas = np.array(df)
        self.datas = preprocessing.scale(datas)
        self.nbr_of_genes = self.datas.shape[1]
        adjacence_df = adjacence_df.replace({'G': ''}, regex=True)
        adjacence_df[0] = adjacence_df[0].astype(int)
        adjacence_df[1] = adjacence_df[1].astype(int)
        self.adjacence_matrix = np.zeros((self.nbr_of_genes, self.nbr_of_genes))
        for index, row in adjacence_df.iterrows():
            if row[2] == 1:
                self.adjacence_matrix[row[0]-1][row[1]-1] = 1

    def get_full_dataset(self):
        return self.datas

    def get_gene_set(self, gene_nbr):
        if gene_nbr >= self.nbr_of_genes:
            #print ' Not enough genes in the data '
            return -1
        train_datas = np.array([el[np.arange(self.datas.shape[1]) != gene_nbr] for el in self.datas])
        answers = np.array([el[gene_nbr] for el in self.datas])
        answers = np.reshape(answers, (-1,1))
        return train_datas, answers

    def get_adjacence_matrix(self):
        return self.adjacence_matrix

