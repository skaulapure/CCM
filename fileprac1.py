import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import quandl as qn

def get_data(list):
    mydata = qn.get(list[1], start_date=list[2], end_date=list[3])
                       #transformation='diff')  # gives you pandas dataframe directly
    return mydata

def dis_plot(df,nm,i):
    plt.figure(i)
    sns.kdeplot(np.log(df.Settle / df.Settle.shift()))
    plt.title(nm)
    return

def main():
    dict1 = {0:['Crude Oil', 'CHRIS/CME_CL2.6' , '2017-01-03', '2017-02-03'],1:['Soybeans', 'CHRIS/CME_S2.6', '2017-01-03', '2017-02-03'],\
             2:['Gold', 'CHRIS/CME_GC2.6', '2017-01-03', '2017-02-03']}

    for i in range (0,len(dict1)):
        df = get_data(dict1[i])
        dis_plot(df,dict1[i][0],i)
        #print (df)
    plt.show()

    return

if __name__ == "__main__":
    main()
