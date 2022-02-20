import pandas as pd
import seaborn as sns
from benchmark_demo.Benchmark import Benchmark
import pickle
import numpy as np

class ResultsInterpreter:
    
    def __init__(self, a_benchmark):
        if isinstance(a_benchmark,Benchmark):
            self.results = a_benchmark.get_results_as_df()
        else:
            raise ValueError("Input should be a Benchmark.\n")

        self.benchmark = a_benchmark
        self.task = a_benchmark.task
        self.methods_ids = a_benchmark.methods_ids
        self.N = a_benchmark.N
        self.repetitions = a_benchmark.repetitions
        self.snr_values = a_benchmark.SNRin
        self.signal_ids = a_benchmark.signal_ids
        self.methods_and_params_dic  = a_benchmark.methods_and_params_dic


        # self.parameters  # TODO parameters collecting for each method.


    def rearrange_data_frame(self, results):
        df = self.get_results_as_df()
        aux_dic = dict()
        new_columns = df.columns.values[0:5].copy()
        new_columns[-1] = 'SNRout'
        for i in range(4,df.shape[1]):
            idx = [j for j in range(4)]+[i]
            df_aux = df.iloc[:,idx]
            df_aux.columns = new_columns
            aux_dic[df.columns[i]] = df_aux
            
        df3 = pd.concat(aux_dic,axis = 0)
        df3 = df3.reset_index()
        df3 = df3.drop('level_1', 1)
        df3.columns.values[0] = 'SNRin'
        return df3


    def write_to_file(self, filename = None):
        if filename is None:
            filename = 'results'

        df = self.benchmark.get_results_as_df()
        column_names = ['Method + Param'] + [col for col in df.columns.values[4::]]
        output_string = ''
        for signal_id in self.signal_ids:
            methods_names = list()
            snr_out_values = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic = dict()

            df2 = df[df['Signal_id']==signal_id]
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2['Method']==metodo]
                if len(self.methods_and_params_dic[metodo])>1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag+'+'+params)
                        valores = df2[(df2['Method']==metodo)&(df2['Parameter']==params)]
                        valores = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                        valores.resize((1,valores.shape[0]))
                        snr_out_values = np.concatenate((snr_out_values,valores))
                else:
                    methods_names.append(tag)
                    valores = df2[df2['Method']==metodo]
                    valores = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                    valores.resize((1,valores.shape[0]))
                    snr_out_values = np.concatenate((snr_out_values,valores))

            snr_out_values = snr_out_values[1::]
            aux_dic[column_names[0]] = methods_names 
            for i in range(1,len(column_names)):
                aux_dic['SNRin: '+ str(column_names[i]) + 'dB'] = snr_out_values[:,i-1]

            df_means = pd.DataFrame(aux_dic)
            # print(signal_id)
            # print(df_means.to_markdown())
            aux_string = '### Signal: '+ signal_id +'\n'+df_means.to_markdown() + '\n'
            output_string += aux_string
            with open(filename + '.md', 'a') as f:
                f.write(aux_string)
                # f.write('/### Signal: '+ signal_id +'\n'+df_means.to_markdown() + '\n')
                # f.write(df_means.to_markdown() + '\n')

        return output_string

