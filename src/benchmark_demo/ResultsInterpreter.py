import pandas as pd
import seaborn as sns
from benchmark_demo.Benchmark import Benchmark
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
import string
import os
import plotly.express as px
import plotly.io as pio


class ResultsInterpreter:
    """This class takes a Benchmark-class object to produce a series of plots and tables
    summarizing the obtained results:
        
    Methods
    -------
    def get_benchmark_as_data_frame(self):
    
    """
    
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
        self.path_results = os.path.join('results')
        self.path_results_figures = os.path.join('results', 'figures')

        # self.parameters  

    def get_benchmark_as_data_frame(self):
        """Returns a DataFrame with the raw data produced by the benchmark with the 
        following format:
            -------------------------------------------------------------------------
            | Method | Parameter | Signal_id | Repetition | SNRin_1 | ... | SNRin_n |
            -------------------------------------------------------------------------

        Returns:
            DataFrame: Raw data of the comparisons. 
        """

        return self.benchmark.get_results_as_df()


    def rearrange_data_frame(self, results=None):
        """Rearrange DataFrame table for using seaborn library. 

        Args:
            results (DataFrame, optional): If not None, must receive the DataFrame 
            produced by a Benchmark-class object using get_results_as_df(). If None,
            uses the Benchmark object given to the constructor of the Interpreter. 
            Defaults to None.

        Returns:
            DataFrame: Rearranged DataFrame
        """

        df = self.benchmark.get_results_as_df()
        aux_dic = dict()
        new_columns = df.columns.values[0:5].copy()
        new_columns[-1] = 'QRF'
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


    def get_df_means(self):
        """ Generates a DataFrame of mean results to .md file. 

        Returns:
            str: String containing the table.
        """
        # if filename is None:
            # filename = 'results'

        df = self.benchmark.get_results_as_df()
        column_names = ['Method'] + [col for col in df.columns.values[4::]]
        output_string = ''
        df_means = list()

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
                        methods_names.append(tag + params)
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
                aux_dic[str(column_names[i])] = snr_out_values[:,i-1]

            df_means.append(pd.DataFrame(aux_dic))

        return df_means


    
    def get_table_means(self):
        """ Generates a table of mean results to .md file. 

        Returns:
            str: String containing the table.
        """
        # if filename is None:
            # filename = 'results'

        df = self.benchmark.get_results_as_df()
        column_names = ['Method + Param'] + [col for col in df.columns.values[4::]]
        output_string = ''
        for signal_id in self.signal_ids:
            methods_names = list()
            snr_out_values = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic = dict()

            # Generate DataFrame with only signal information
            df2 = df[df['Signal_id']==signal_id]
            # Save .csv file.
            filename = os.path.join('results','results_'+signal_id+'.csv')
            df2.to_csv(filename)


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
                # aux_dic['SNRin: '+ str(column_names[i]) + 'dB'] = snr_out_values[:,i-1]
                aux_dic[str(column_names[i])] = snr_out_values[:,i-1]

            df_means = pd.DataFrame(aux_dic)
            # print(df_means)
            # print(signal_id)
            # print(df_means.to_markdown())
            aux_string = '### Signal: '+ signal_id + '  [[View Plot]]('+ 'plot_'+signal_id+'.html)  '+'  [[Get .csv]]('+ 'results_'+signal_id+'.csv)' +'\n'+ df_means.to_markdown() + '\n'
            output_string += aux_string

            filename = os.path.join('results','plot_'+signal_id+'.html')
            with open(filename, 'w') as f:
                df3 = df_means.set_index('Method + Param').stack().reset_index()
                # print(df3)
                df3.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
                fig = px.line(df3, x="SNRin", y="QRF", color='Method + Param', markers=True)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        return output_string


    def save_report(self, filename = None):
        """ This function generates a report of the results given in the Benchmark-class
        object. The report is saved in a MardkedDown syntax to be viewed as a .md file.

        Args:
            filename (str, optional): Path for saving the report. Defaults to None.
        """

        self.get_summary_grid()

        lines = ['# Benchmark Report' +'\n',
                '## Configuration' + '   [Get .csv file] ' + '\n',
                # 'Parallelize' + str(self.benchmark.parallel_flag) + '\n',
                'Length of signals: ' + str(self.N) + '\n', 
                'Repetitions: '+ str(self.repetitions) + '\n',
                'SNRin values: ']

        lines = lines + [str(val) + ', ' for val in self.snr_values] + ['\n']
        lines = lines + ['### Methods  \n'] + ['* ' + methid +' \n' for methid in self.methods_ids]
        lines = lines + ['### Signals  \n'] + ['* ' + signid +' \n' for signid in self.signal_ids]
        # lines = lines + ['## Figures:\n ![Summary of results](results/../figures/plots_grid.png) \n'] 
        lines = lines + ['## Mean results tables: \n']
       
        if filename is None:
            filename = os.path.join('results','readme.md')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
            # f.writelines(lines)

        output_string = self.get_table_means()

        with open(filename, 'a') as f:
          f.write(output_string)

        # last_line = '|--------------------------------------------------------------------:|'
        
        # with open(filename, 'a') as f:
        #     f.write('\n'.join(last_line))

    def get_snr_plot(self, df, x=None, y=None, hue=None, axis = None):
        """ Generates a Quality Reconstruction Factor (QRF) vs. SNRin plot. The QRF is 
        computed as: 
        QRF = 20 log ( norm(x) / norm(x-x_r)) [dB]
        where x is the noiseless signal and x_r is de denoised estimation of x.

        Args:
            df (DataFrame): DataFrame with the results of the simulation.
            x (str, optional): Column name to use as the horizontal axis. 
            Defaults to None.
            y (str, optional): Column name to use as the vertical axis. 
            Defaults to None.
            hue (str, optional): Column name with the methods' name. Defaults to None.
            axis (matplotlib.Axes, optional): The axis object where the plot will be 
            generated. Defaults to None.
        """

        markers = ['o','d','s','*']
        aux = np.unique(df[hue].to_numpy())
        # print(aux)
        # fig, axis2 = plt.subplots(1,1)
        
        plots = [(method_name, markers[np.mod(i,4)]) for i, method_name in enumerate(aux)]
        u_offset = np.linspace(-2,2,len(plots))
        u_offset = np.zeros_like(u_offset)
        for offs_idx, plots_info in enumerate(plots):
            method_name, marker = plots_info 
            df_aux = df[df[hue]==method_name]
            u = np.unique(df_aux[x].to_numpy())
            v = np.zeros_like(u)
            
            label = ''.join([c for c in string.capwords(method_name, sep = '_') if c.isupper()])
            if method_name.find('-') > -1:
                label= label+method_name[method_name.find('-')::]

            for uind, j in enumerate(u):
                df_aux2 = df_aux[df_aux[x]==j]
                no_nans = df_aux2[y].dropna()
                if no_nans.shape != (0,):
                    v[uind] = np.nanmean(no_nans.to_numpy())

            axis.plot(u+u_offset[offs_idx],v,'-'+ marker, ms = 5, linewidth = 1.0, label=label)
            
        # axis.plot([np.min(u), np.max(u)],[np.min(u), np.max(u)],'r',
                                            # linestyle = (0, (5, 10)),
                                            # linewidth = 0.75)
        axis.set_xticks(u)
        axis.set_yticks(u)
        axis.set_xlabel(x + ' (dB)')
        axis.set_ylabel(y + ' (dB)')
        return

    def get_snr_plot2(self, df, x=None, y=None, hue=None, axis = None):
        """ Generates a Quality Reconstruction Factor (QRF) vs. SNRin plot. The QRF is 
        computed as: 
        QRF = 20 log ( norm(x) / norm(x-x_r)) [dB]
        where x is the noiseless signal and x_r is de denoised estimation of x.

        Args:
            df (DataFrame): DataFrame with the results of the simulation.
            x (str, optional): Column name to use as the horizontal axis. 
            Defaults to None.
            y (str, optional): Column name to use as the vertical axis. 
            Defaults to None.
            hue (str, optional): Column name with the methods' name. Defaults to None.
            axis (matplotlib.Axes, optional): The axis object where the plot will be 
            generated. Defaults to None.
        """

        markers = ['o','d','s','*']
        line_style = ['--' for i in self.methods_ids]
        sns.pointplot(x="SNRin", y="QRF", hue="Method",
                    capsize=0.15, height=10, aspect=1.0, dodge=0.5,
                    kind="point", data=df, errwidth = 0.7,
                    ax = axis) #linestyles=line_style,
            
            # axis.set_xticks(u)
            # axis.set_yticks(u)
            # axis.set_xlabel(x + ' (dB)')
            # axis.set_ylabel(y + ' (dB)')
        
    def get_snr_plot_bars(self, df, x=None, y=None, hue=None, axis = None):
        """ Generates a Quality Reconstruction Factor (QRF) vs. SNRin plot. The QRF is 
        computed as: 
        QRF = 20 log ( norm(x) / norm(x-x_r)) [dB]
        where x is the noiseless signal and x_r is de denoised estimation of x.

        Args:
            df (DataFrame): DataFrame with the results of the simulation.
            x (str, optional): Column name to use as the horizontal axis. 
            Defaults to None.
            y (str, optional): Column name to use as the vertical axis. 
            Defaults to None.
            hue (str, optional): Column name with the methods' name. Defaults to None.
            axis (matplotlib.Axes, optional): The axis object where the plot will be 
            generated. Defaults to None.
        """

        markers = ['o','d','s','*']
        line_style = ['--' for i in self.methods_ids]
        sns.barplot(x="SNRin", y="QRF", hue="Method",
                    data=df, errwidth = 0.7,
                    ax = axis)
            
            # axis.set_xticks(u)
            # axis.set_yticks(u)
            # axis.set_xlabel(x + ' (dB)')
            # axis.set_ylabel(y + ' (dB)')    


    def get_summary_grid(self, filename = None, savetofile=True):
        """ Generates a grid of QRF plots for each signal, displaying the performance 
        of all methods for all noise conditions.

        Args:
            size (tuple, optional): Size of the figure in inches. Defaults to (3,3).

        Returns:
            Matplotlib.Figure: Returns a figure handle.
        """
        
        Nsignals = len(self.signal_ids)
        df_rearr = self.rearrange_data_frame()
        sns.set(style="ticks", rc={"lines.linewidth": 0.7})
        
        if Nsignals < 4:
            nrows_ncols=(1,Nsignals)
        else:
            nrows_ncols=(int(np.ceil(Nsignals/4)),4)

        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0)


        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                 nrows_ncols=nrows_ncols,  # creates 2x2 grid of axes
        #                 axes_pad=0.5,  # pad between axes in inch.
        #                 )

        # grid = gridspec.GridSpec(nrows_ncols[0], nrows_ncols[1],
                                    #  width_ratios=[1 for i in range(nrows_ncols[1])],
                                    #  sharex=True)

        fig, grid = plt.subplots(nrows_ncols[0], nrows_ncols[1], constrained_layout=False, sharex=True, sharey=True)
        
        for signal_id, ax in zip(self.signal_ids, grid):
            # sns.set_theme()
            # ax = fig.add_subplot(subplot) 
            df_aux = df_rearr[df_rearr['Signal_id']==signal_id]
            indexes = df_aux['Parameter']!='None'
            df_aux.loc[indexes,'Method'] = df_aux.loc[indexes,'Method'] +'-'+ df_aux.loc[indexes,'Parameter']  
            # print(df_aux)

            # self.get_snr_plot(df_aux, x='SNRin', y='QRF', hue='Method', axis = ax)
            # self.get_snr_plot2(df_aux, x='SNRin', y='SNRout', hue='Method', axis = ax)
            self.get_snr_plot_bars(df_aux, x='SNRin', y='SNRout', hue='Method', axis = ax)
            ax.grid(linewidth = 0.5)
            ax.set_title(signal_id)
            # ax.set_box_aspect(1)
            # sns.despine(offset=10, trim=True)
            ax.legend([],[], frameon=False)
            ax.legend(loc='upper left', frameon=False, fontsize = 'xx-small')
            

        fig.set_size_inches((12,4*Nsignals//4))
        
        if filename is None:
            filename = os.path.join('results','figures','plots_grid.png')

        if savetofile:
            fig.savefig(filename,bbox_inches='tight')# , format='svg')
    
        return fig


    def get_summary_plots(self, size=(3,3), savetofile=True, filename=None):
        """ Generates individual QRF plots for each signal, displaying the performance 
        of all methods for all noise conditions.

        Args:
            size (tuple, optional): Size of the figure in inches. Defaults to (3,3).

        Returns:
            Matplotlib.Figure: Returns a figure handle.
        """
        
        Nsignals = len(self.signal_ids)
        df_rearr = self.rearrange_data_frame()
        list_figs = list()

        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #                 nrows_ncols=(3,Nsignals//3),  # creates 2x2 grid of axes
        #                 axes_pad=0.5,  # pad between axes in inch.
        #                 )
        
        for signal_id in self.signal_ids:
            fig,ax = plt.subplots(1,1)
            print(signal_id) 
            # sns.set_theme() 
            df_aux = df_rearr[df_rearr['Signal_id']==signal_id]
            indexes = df_aux['Parameter']!='None'
            df_aux.loc[indexes,'Method'] = df_aux.loc[indexes,'Method'] +'-'+ df_aux.loc[indexes,'Parameter']  
            self.get_snr_plot(df_aux, x='SNRin', y='QRF', hue='Method', axis = ax)
            ax.grid(linewidth = 0.5)
            ax.set_title(signal_id)
            ax.legend(loc='upper left', frameon=False, fontsize = 'small')
            # sns.despine(offset=10, trim=True)
            fig.set_size_inches(size)
            
            if savetofile:
                if filename is None:
                    fig.savefig('results/figures/plot_'+ signal_id +'.pdf',
                                bbox_inches='tight')# , format='svg')
                else:
                    fig.savefig(filename + signal_id +'.pdf',
                                bbox_inches='tight')# , format='svg')

            list_figs.append(fig)
        return list_figs

    def save_csv_files(self):
        """Save results in .csv files.

        Args:
            filepath (str, optional): Path to file. Defaults to None.

        Returns:
            bool: True if the file was saved.
        """
        
        df1 = self.get_benchmark_as_data_frame()
        df2 = self.rearrange_data_frame()

        filename1 = os.path.join('results','denoising_results_raw.csv')
        filename2 = os.path.join('results','denoising_results_rearranged.csv')
        df1.to_csv(filename1)
        df2.to_csv(filename2)

        return True

    def rank_the_methods(self):
        return True

    def save_html_plots(self):
        df_means  = self.get_df_means()
        df = df_means[0]
        df = df.set_index('Method').stack().reset_index()
        df.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
        with open('p_graph.html', 'a') as f:
            for df in df_means:
                df = df.set_index('Method').stack().reset_index()
                df.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
                fig = px.line(df, x="SNRin", y="QRF", color='Method', markers=True,width=200, height=200)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))