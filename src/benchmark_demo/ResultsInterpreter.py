from sqlite3 import DatabaseError
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
            snr_out_mean = np.zeros((1, len([col for col in df.columns.values[4::]])))
            snr_out_std = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic = dict()

            df2 = df[df['Signal_id']==signal_id]
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2['Method']==metodo]
                if len(self.methods_and_params_dic[metodo])>1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag + params)
                        valores = df2[(df2['Method']==metodo)&(df2['Parameter']==params)]
                        # Compute the means
                        valores_mean = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                        valores_mean.resize((1,valores_mean.shape[0]))
                        snr_out_mean = np.concatenate((snr_out_mean,valores_mean))
                        # Compute the std
                        valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                        valores_std.resize((1,valores_std.shape[0]))
                        snr_out_std = np.concatenate((snr_out_std,valores_std))

                else:
                    methods_names.append(tag)
                    valores = df2[df2['Method']==metodo]
                    # Compute the means
                    valores_mean = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                    valores_mean.resize((1,valores_mean.shape[0]))
                    snr_out_mean = np.concatenate((snr_out_mean,valores_mean))
                    # Compute the std
                    valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                    valores_std.resize((1,valores_std.shape[0]))
                    snr_out_std = np.concatenate((snr_out_std,valores_std))

            snr_out_mean = snr_out_mean[1::]
            snr_out_std = snr_out_std[1::]
            aux_dic[column_names[0]] = methods_names 
            for i in range(1,len(column_names)):
                aux_dic[str(column_names[i])] = snr_out_mean[:,i-1]

            df_means.append(pd.DataFrame(aux_dic))

        return df_means

    def get_df_std(self):
        """ Generates a DataFrame of std results to .md file. 

        Returns:
            str: String containing the table.
        """
        # if filename is None:
            # filename = 'results'

        df = self.benchmark.get_results_as_df()
        column_names = ['Method'] + [col for col in df.columns.values[4::]]
        output_string = ''
        df_std = list()

        for signal_id in self.signal_ids:
            methods_names = list()
            snr_out_std = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic = dict()

            df2 = df[df['Signal_id']==signal_id]
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2['Method']==metodo]
                if len(self.methods_and_params_dic[metodo])>1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag + params)
                        valores = df2[(df2['Method']==metodo)&(df2['Parameter']==params)]
                        # Compute the std
                        valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                        valores_std.resize((1,valores_std.shape[0]))
                        snr_out_std = np.concatenate((snr_out_std,valores_std))

                else:
                    methods_names.append(tag)
                    valores = df2[df2['Method']==metodo]
                    # Compute the std
                    valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                    valores_std.resize((1,valores_std.shape[0]))
                    snr_out_std = np.concatenate((snr_out_std,valores_std))

            snr_out_std = snr_out_std[1::]
            aux_dic[column_names[0]] = methods_names 
            for i in range(1,len(column_names)):
                aux_dic[str(column_names[i])] = snr_out_std[:,i-1]

            df_std.append(pd.DataFrame(aux_dic))

        return df_std



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
            snr_out_values_std = np.zeros((1, len([col for col in df.columns.values[4::]])))
            aux_dic_mean = dict()
            aux_dic_std = dict()

            # Generate DataFrame with only signal information
            df2 = df[df['Signal_id']==signal_id]
            
            # Save .csv file for the signal.
            csv_filename = os.path.join('results',self.task,'csv_files','results_'+signal_id+'.csv')
            df2.to_csv(csv_filename)

            # For each method, generates the mean and std of results, and get figures.
            for metodo in self.methods_and_params_dic:
                tag = metodo
                aux = df2[df2['Method']==metodo]
                if len(self.methods_and_params_dic[metodo])>1:
                    for params in self.methods_and_params_dic[metodo]:
                        methods_names.append(tag+'+'+params)
                        valores = df2[(df2['Method']==metodo)&(df2['Parameter']==params)]
                        # Computing mean
                        valores_mean = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                        valores_mean.resize((1,valores_mean.shape[0]))
                        snr_out_values = np.concatenate((snr_out_values,valores_mean))
                        # Computing std
                        valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                        valores_std.resize((1,valores_std.shape[0]))
                        snr_out_values_std = np.concatenate((snr_out_values_std,valores_std))
                else:
                    methods_names.append(tag)
                    valores = df2[df2['Method']==metodo]
                    # Computing mean
                    valores_mean = valores.iloc[:,4::].to_numpy().mean(axis = 0)
                    valores_mean.resize((1,valores_mean.shape[0]))
                    snr_out_values = np.concatenate((snr_out_values,valores_mean))
                    # Computing std
                    valores_std = valores.iloc[:,4::].to_numpy().std(axis = 0)
                    valores_std.resize((1,valores_std.shape[0]))
                    snr_out_values_std = np.concatenate((snr_out_values_std,valores_std))

            snr_out_values = snr_out_values[1::]
            snr_out_values_std = snr_out_values_std[1::]
            aux_dic_mean[column_names[0]] = methods_names
            aux_dic_std[column_names[0]] = methods_names  
            for i in range(1,len(column_names)):
                # aux_dic['SNRin: '+ str(column_names[i]) + 'dB'] = snr_out_values[:,i-1]
                aux_dic_mean[str(column_names[i])] = snr_out_values[:,i-1]
                aux_dic_std[str(column_names[i])] = snr_out_values_std[:,i-1]

            # Generate DataFrames for plotting easily
            df_means = pd.DataFrame(aux_dic_mean)
            df_means_aux = df_means.copy()
            df_std = pd.DataFrame(aux_dic_std)

            # Check maxima to highlight:
            nparray_aux = df_means.iloc[:,1::].to_numpy()
            maxinds = np.argmax(nparray_aux, axis=0)

            for col, max_ind in enumerate(maxinds):
                df_means_aux.iloc[max_ind,col+1] =  '**' + str(df_means.iloc[max_ind,col+1]) + '**'        


            # Change column names to make it more human-readable
            df_results = pd.DataFrame()
            df_results[column_names[0]] = df_means[column_names[0]]
            for col_ind in range(1,len(column_names)):
                # print(column_names[col_ind])
                # df_aux = pd.DataFrame()
                # df_aux['QRF (mean)'] = df_means[str(column_names[col_ind])]
                # df_aux['QRF (sd)'] = df_std[str(column_names[col_ind])]
                # ddd['SNRin='+str(column_names[col_ind])+'dB (mean)'] = df_aux
                df_results['SNRin='+str(column_names[col_ind])+'dB (mean)'] = df_means_aux[str(column_names[col_ind])]
                df_results['SNRin='+str(column_names[col_ind])+'dB (std)'] = df_std[str(column_names[col_ind])]


            # Table header with links
            csv_filename = os.path.join('.',self.task,'csv_files','results_'+signal_id+'.csv')
            aux_string = '### Signal: '+ signal_id + '  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/'+ 'plot_'+signal_id+'.html)  '+'  [[Get .csv]](/results/denoising/csv_files/results_' + signal_id +'.csv' +')' +'\n'+ df_results.to_markdown() + '\n'
            output_string += aux_string

            # Generate .html interactive plots files with plotly
            html_filename = os.path.join('results',self.task,'figures','html','plot_'+signal_id+'.html')
            with open(html_filename, 'w') as f:
                df3 = df_means.set_index('Method + Param').stack().reset_index()
                df3.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
                df3_std = df_std.set_index('Method + Param').stack().reset_index()
                df3_std.rename(columns = {'level_1':'SNRin', 0:'std'}, inplace = True)
                df3['std'] = df3_std['std']
                # print(df3)
                fig = px.line(df3, x="SNRin", y="QRF", color='Method + Param', markers=True, error_x = "SNRin", error_y = "std")
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        return output_string


    def save_report(self, filename = None):
        """ This function generates a report of the results given in the Benchmark-class
        object. The report is saved in a MardkedDown syntax to be viewed as a .md file.

        Args:
            filename (str, optional): Path for saving the report. Defaults to None.
        """

        # self.get_summary_grid()

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

        if self.task == "denoising":
            lines = lines + ['Results shown here are the mean and standard deviation of \
                              the Quality Reconstruction Factor. \
                              Best performances are **bolded**. \n']
        
        if self.task == "detection":
            lines = lines + ['Results shown here are the mean and standard deviation of \
                            the estimated detection power. \
                            Best performances are **bolded**. \n']
       


        if filename is None:
            filename = os.path.join('results','results_'+self.task+'.md')

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
            # f.writelines(lines)

        output_string = self.get_table_means()
        
        self.save_csv_files()

        with open(filename, 'a') as f:
          f.write(output_string)

    
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

            self.get_snr_plot(df_aux, x='SNRin', y='QRF', hue='Method', axis = ax)
            # self.get_snr_plot2(df_aux, x='SNRin', y='SNRout', hue='Method', axis = ax)
            # self.get_snr_plot_bars(df_aux, x='SNRin', y='SNRout', hue='Method', axis = ax)
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


    def get_summary_plots(self, 
                        size=(3,3), 
                        savetofile=True, 
                        filename=None, 
                        plot_type='lines'):
                        
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
            df_aux.loc[indexes,'Method'] = df_aux.loc[indexes,'Method']+'-'+ df_aux.loc[indexes,'Parameter']

            if plot_type == 'lines':
                self.get_snr_plot(df_aux, x='SNRin', y='QRF', hue='Method', axis = ax)


            if plot_type == 'bars':
                self.get_snr_plot_bars(df_aux, x='SNRin', y='QRF', hue='Method', axis = ax)

            if self.benchmark.task == "detection":
                ax.set_ylabel('Detection Power')
                ax.set_ylim([0, 2])

            ax.grid(linewidth = 0.5)
            ax.set_axisbelow(True)
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

        filename1 = os.path.join('results',self.task,'csv_files','denoising_results_raw.csv')
        filename2 = os.path.join('results',self.task,'csv_files','denoising_results_rearranged.csv')
        df1.to_csv(filename1)
        df2.to_csv(filename2)

        return True

    # def save_html_plots(self):
    #     df_means  = self.get_df_means()
    #     df_std = self.get_df_std()
    #     # df = df_means[0]
    #     # df = df.set_index('Method').stack().reset_index()
    #     # df.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
    #     with open('p_graph.html', 'a') as f:
    #         for df, dfs in zip(df_means,df_std):
    #             df = df.set_index('Method').stack().reset_index()
    #             df.rename(columns = {'level_1':'SNRin', 0:'QRF'}, inplace = True)
    #             dfs = df_std[0].set_index('Method').stack().reset_index()
    #             dfs.rename(columns = {'level_1':'SNRin', 0:'std'}, inplace = True)
                
    #             df['std'] = dfs['std']
    #             fig = px.line(df, x="SNRin", y="QRF", color='Method', markers=True)
    #             f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))