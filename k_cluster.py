"""
    Title: k-cluster.py
    Author: Clayton Broman
    Created on: 02/16/22.
    Description: Takes as an input a CSV file location. It has only been tested with Wisconsin Breast Cancer Diognostic Training set

    Purpose: Use K-clustsering Algorithm to group data.
    Usage:  see __main__ at bottom for example code
    returns: plots of data, malignant as red, benign as black
    Build with:  python k_cluster.py

    This code was written including many code snippets and concepts borrowed from
    comprihensive k-clustering guide
    https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
    License        : GNU public
"""
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class KCluster:
    def __init__(self, file = None, names_n = None):
        self.K = 2
        self.data = None
        self.names_n = names_n
        if file:
            self.read_file(file, names_n)

    def read_file(self, file:str, names_n):
        self.data = pd.read_csv(file,index_col=False,header=None, names= names_n)

    def plot_raw_data(self, malig, benign):
        #Plot all the columns of the data against each other
        for i in range(1,len(self.names_n),1):
            x = self.names_n[i]
            for j in range(i+1,len(self.names_n),1):
                y = self.names_n[j]
                X_mal = malig[[x,y]]
                X_ben = benign[[x,y]]
                #Visualise data points
                plt.scatter(X_mal[x],X_mal[y],c='red')
                plt.scatter(X_ben[x],X_ben[y],c='black')
                #plt.imshow(X, cmap='hot', interpolation='nearest')
                plt.xlabel(x)
                plt.ylabel(f'{y}')
                plt.show()

    def cluster(self, col1, col2):
        # cluster data points from col1 and col2 and display them
        N = self.data.shape[0]
        #print(N)
        K = 2 # number of clusters
        x = col1
        y = col2
        X = self.data[[x,y]]
        centroids = (X.sample(n = K))  # need to do use weighted averages to compute not rand
        #plt.scatter(X[x],X[y],c='black')
        #plt.scatter(centroids[x],centroids[y],c='green')
        #plt.xlabel(x)
        #plt.ylabel(y)
        #plt.show()

        # Step 3 - Assign all the points to the closest cluster centroid
        # Step 4 - Recompute centroids of newly formed clusters
        # Step 5 - Repeat step 3 and 4

        diff = 1
        j=0
        import copy

        while(diff!=0):
            XD=copy.deepcopy(X)
            i=1
            # for each centroid
            for index1,row_c in centroids.iterrows():
                ED=[]
                #measure distance from centroid to every point
                for index2,row_d in XD.iterrows():
                    d1=(row_c[x]-row_d[x])**2
                    d2=(row_c[y]-row_d[y])**2
                    d=np.sqrt(d1+d2)
                    ED.append(d)

                X[i]=ED # store distance vector as part of original Matrix
                i=i+1

            C=[]
            #find nearest centroid for each point
            for index,row in X.iterrows():
                min_dist=row[1]
                pos=1
                for i in range(K):
                    if row[i+1] < min_dist:
                        min_dist = row[i+1]
                        pos=i+1
                C.append(pos)
            X["Cluster"]=C
            Centroids_new = X.groupby(["Cluster"]).mean()[[y,x]]
            if j == 0:
                diff=1
                j=j+1
            else:
                diff = (Centroids_new[y] - centroids[y]).sum() + (Centroids_new[x] - centroids[x]).sum()
                print(diff.sum())
            centroids = X.groupby(["Cluster"]).mean()[[y,x]]

        color=['black','red','cyan']
        for k in range(K):
            data=X[X["Cluster"]==k+1]
            plt.scatter(data[x],data[y],c=color[k])
        plt.scatter(centroids[x],centroids[y],c='green')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

if __name__ == "__main__":
    # names = ['id_num','thickness','uni_size','uni_shape','adhesion','epi_size','bare_nuclei','chromatin','nucleoli','mitoses','class']
    names_n = ['id_num', 'outcome', 'rad', 'texture', 'perim', 'area', 'smooth', 'compact', 'concave', 'concave_points',
               'sym', 'fractal_dim', \
               'rad_SE', 'texture_SE', 'perim_SE', 'area_SE', 'smooth_SE', 'compact_SE', 'concave_SE',
               'concave_points_SE', 'sym_SE', 'fractal_dim_SE', \
               'rad_worst', 'texture_worst', 'perim_worst', 'area_worst', 'smooth_worst', 'compact_worst',
               'concave_worst', 'concave_points_worst', 'sym_worst', 'fractal_dim_worst']
    file = '/Users/claybro/Documents/Personal/CSCI499/wdbc.data'

    kc = KCluster(file, names_n)
    #convert M and R to 0,1
    print(kc.data)
    kc.data['outcome'] = kc.data['outcome'].map(lambda diag: bool(diag == "M"))  # M being cancerous
    # break into two dataframes malignant and benign
    malig = kc.data[kc.data.outcome == True]
    # print(malig.outcome)
    benign = kc.data[kc.data.outcome == False]
    #try to see some correlations
    #kc.plot_raw_data(malig, benign)
    kc.cluster('rad', 'compact')

