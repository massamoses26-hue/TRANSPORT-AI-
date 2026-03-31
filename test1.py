from sklearn.datasets import make_blobs
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt




def create_H_V_D_T(V,H,D):
    len_V=len(V)
    len_H=len(H)
    len_D=len(D)
    HVD={}

    for i in range(len_H):
        for j in range(len_V):
            for k in range(len_D):
                HVD['H'+str(i+1)+'_V'+str(j+1)+'_D'+str(k+1)]=[H['H'+str(i+1)],V['V'+str(j+1)],D['D'+str(k+1)]]

    HVDT={}
    L_VH=['H1','H2','H3','H4','H5']
    DeltaTime=[[0,0],[1.5,2],[0,0.5],[2,3],[0,0]]
    
    for j in range(len(L_VH)):
        for i in HVD:
            if i[:2]==L_VH[j]:
                
                D_HVD= [HVD[i][2][0],HVD[i][2][-1]]
                D_HVD
                V_HVD= [HVD[i][1][0],HVD[i][1][-1]]
                V_HVD
                T_HVD= [(D_HVD[0]/V_HVD[0])+DeltaTime[j][0],(D_HVD[1]/V_HVD[1])+DeltaTime[j][1]]
                T_HVD
                T=np.linspace(T_HVD[0],T_HVD[1],50)
                
                HVD[i].append(T)
                HVDT[i+'_T']= HVD[i]

                
    return HVDT




def create_pandas_dataset(V,H,D):

    import pandas as pd
    HVDT= create_H_V_D_T(V,H,D)
    
    df= pd.DataFrame({
        'Heure_Depart':[],
        'Vitesse_Moyenne':[],
        'Distance':[],
        'Temps_Estimé':[]
    })
    
    for i in HVDT:
        
        H=[]
        V=[]
        D=[]
        T=[]

        for j in HVDT[i][0]:
            H.append(float(j))
        for j in HVDT[i][1]:
            V.append(float(j))
        for j in HVDT[i][2]:
            D.append(float(j))
        for j in HVDT[i][3]:
            T.append(float(j))
            
        datasets = pd.DataFrame({'Heure_Depart':H})
        datasets['Vitesse_Moyenne'] = pd.DataFrame({'Vitesse_Moyenne':V})
        datasets['Distance'] = pd.DataFrame({'Distance':D})
        datasets['Temps_Estimé'] = pd.DataFrame({'Temps_Estimé':T})

        df= pd.concat([df, datasets], ignore_index=True)

    df.to_csv('Transport_Set.csv', index=False)

    return df




def transport_LinearRegression_modal(df_file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    df= pd.read_csv(df_file)
    X=df[['Heure_Depart','Vitesse_Moyenne','Distance']].values
    y=df[['Temps_Estimé']].values


    X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    modal= LinearRegression()
    modal.fit(X_train,y_train)

    prediction_train=modal.predict(X_train)
    prediction_test=modal.predict(X_test)
    
    score_train=modal.score(X_train,y_train)
    score_test=modal.score(X_test,y_test)
    
    print("Score de l'entrainement : ", score_train)
    print("Score du test : ", score_test)

    dict_train={
        'X':X_train,
        'y':y_train,
        'prediction':prediction_train,
        'score':score_train
    }
    
    dict_test={
        'X':X_test,
        'y':y_test,
        'prediction':prediction_test,
        'score':score_test
    }

    
    return(modal,dict_train,dict_test)






def transport_RandomForestRegressor_modal(df_file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    df= pd.read_csv(df_file)
    X=df[['Heure_Depart','Vitesse_Moyenne','Distance']].values
    y=df[['Temps_Estimé']].values


    X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    modal= RandomForestRegressor(n_estimators=100, random_state=42)
    modal.fit(X_train,y_train)

    prediction_train=modal.predict(X_train)
    prediction_test=modal.predict(X_test)
    
    score_train=modal.score(X_train,y_train)
    score_test=modal.score(X_test,y_test)
    
    print("Score de l'entrainement : ", score_train)
    print("Score du test : ", score_test)

    dict_train={
        'X':X_train,
        'y':y_train,
        'prediction':prediction_train,
        'score':score_train
    }
    
    dict_test={
        'X':X_test,
        'y':y_test,
        'prediction':prediction_test,
        'score':score_test
    }

    
    return(modal,dict_train,dict_test)

