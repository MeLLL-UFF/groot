import matplotlib.pyplot as plt


class Plots:

	def __init__(self):
		plt.rcParams["font.sans-serif"] = "Ubuntu Condensed"
		plt.style.use('seaborn-whitegrid')


	def make_plot(self, m_data, std_data, image_name, data_name='mean(cll)', title='CLL over the generations'):
	    plt.figure(figsize=(10, 3))
	    plt.xlabel('generations')
	    plt.ylabel(data_name)
	    plt.title(title)
	    positions = list(range(len(m_data)))
	    plt.errorbar(positions, m_data, yerr=std_data,fmt='-o')
	    plt.annotate(f'{(round(m_data[0],2))}', (0+0.1, m_data[0]), arrowprops=dict(facecolor='black', shrink=100.0),
	           ha='left', va='bottom',fontsize=12)
	    plt.annotate(f'{(round(m_data[-1],3))}', (positions[-1]+0.1, m_data[-1]), arrowprops=dict(facecolor='black', shrink=100.0),
	       ha='left', va='bottom',fontsize=12)
	    plt.savefig(f'{image_name}.png')

	def make_many_plots(self, m_data, std_data, image_name, data_name='mean(cll)', title='CLL over the generations'):
	    #pode ser 2 tipos de dados no mesmo gráfico
	    plt.figure(figsize=(10, 3))
	    plt.xlabel('generations')
	    plt.title(title)
	    for _m_data, _s_data, name in zip(m_data, std_data, data_name):
	        positions = list(range(len(_m_data)))
	        plt.errorbar(positions, _m_data, yerr=_s_data,fmt='-o',label=name)
	        plt.annotate(f'{(round(_m_data[0],2))}', (0+0.1, _m_data[0]), arrowprops=dict(facecolor='black', shrink=100.0),
	           ha='left', va='bottom',fontsize=12)
	        plt.annotate(f'{(round(_m_data[-1],2))}+/-{(round(_s_data[-1],4))}', (positions[-1]+0.1, _m_data[-1]), arrowprops=dict(facecolor='black', shrink=100.0),
	           ha='left', va='bottom',fontsize=12)
	    plt.legend()
	    plt.savefig(image_name)
