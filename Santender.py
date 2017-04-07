# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:24:37 2016

@author: vushesh
"""

import pandas as pd
import numpy as np
import xgboost as xgb


mydata = pd.read_csv("D:/Studies/Online Competitions/Santner recommendation/train_ver2.csv")


mydata["age"] = mydata["age"].apply(lambda x: x if isinstance(x,int) else np.nan if x.strip() == 'NA' else int(x.strip()) )
mydata["age"] = np.int64(mydata["age"])

mydata["antiguedad"] = mydata["antiguedad"].apply(lambda x: x if isinstance(x,int) else np.nan if x.strip() == 'NA' else int(x.strip()) )
mydata["antiguedad"] = np.int64(mydata["antiguedad"])

mydata["indrel_1mes"] = mydata["indrel_1mes"].apply(lambda x: 1 if x in ['1.0',1.0] else 2 
                                                    if x in ['2.0',2.0] else 3 if x in ['3.0',3.0] else 4 
                                                    if x in ['4.0',4.0] else 5 )



temp = mydata[["cod_prov","renta"]]
pas = temp.dropna()

a = pas.groupby("cod_prov").median()


mydata = mydata[mydata.fecha_dato != '2016-04-28']


mydata["renta"] = temp.apply(lambda x: a.loc[x["cod_prov"]]["renta"] if np.isnan(x["renta"]) and 
                                                                    str(x["cod_prov"]) != 'nan' else x["renta"], axis=1 )

d = set(mydata.columns) - set(["ult_fec_cli_1t" ,"conyuemp","nomprov"])

 
mydata["canal_entrada"].fillna("KHE",inplace = True )

mydata = mydata[list(d)]   

mydata = mydata.dropna()

mydata["ind_empleado"] = mydata["ind_empleado"].apply(lambda x: 1 if x == "N" else 2 if x == "B" else 3 if x == "F" else 4 if x == "A"
                                                                else 5 )

mydata["sexo"] = mydata["sexo"].apply(lambda x: 0 if x == "V" else 1)

mydata["ind_nuevo"] = mydata["ind_nuevo"].apply(lambda x: 0 if x == '0' else 1)

mydata["indrel"] = mydata["indrel"].apply(lambda x: 1 if x == '0' else 0)

mydata["tiprel_1mes"] = mydata["tiprel_1mes"].apply(lambda x:1 if x == "I" else 2 if x == "A" else 3 if x == "P" else 4 if x == "R"
                                                                else 5 )

mydata["indresi"] = mydata["indresi"].apply(lambda x: 1 if x == "S" else 2)

mydata["indext"] = mydata["indext"].apply(lambda x: 1 if x == "S" else 2)

mydata["indfall"] = mydata["indfall"].apply(lambda x: 1 if x == "S" else 2)

mydata["tipodom"] = mydata["tipodom"].apply(lambda x: 1 if x == "1" else 0)

mydata["ind_actividad_cliente"] = mydata["ind_actividad_cliente"].apply(lambda x:0 if x == "0" else 1)

mydata["segmento"] = mydata["segmento"].apply(lambda x: 0 if x == "02 - PARTICULARES" else 1 if x == "03 - UNIVERSITARIO" else 2)

mydata["cod_prov"] = mydata["cod_prov"].apply(lambda x: int(x))

mydata["ind_nomina_ult1"] = mydata["ind_nomina_ult1"].apply(lambda x: 1 if x == 1.0 else 0)

mydata["ind_nom_pens_ult1"] = mydata["ind_nom_pens_ult1"].apply(lambda x: 1 if x == 1.0 else 0)


mapd = {
'pais_residencia' : {'LV': 113, 'BE': 64, 'BG': 17, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 50, 'BY': 12, 'BZ': 102, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}   

mydata = pd.read_csv("D:/Studies/Online Competitions/Santner recommendation/test_edited.csv")

mydata["pais_residencia"] = mydata["pais_residencia"].apply(lambda x: mapd["pais_residencia"][x])
mydata["canal_entrada"] = mydata["canal_entrada"].apply(lambda x: mapd["canal_entrada"][x])

mydata = pd.read_csv("D:/Studies/Online Competitions/Santner recommendation/edited.csv")
mydata["indrel_1mes"] = mydata["indrel_1mes"].apply(lambda x:2 if x == 2 else 3 if x == 3 else 4 if x == "P" else 1)
mydata.to_csv('edited.csv',header=True,cols=mydata.columns)


data = pd.read_csv("D:/Studies/Online Competitions/Santner recommendation/test_ver2.csv")
data.columns

data["renta"] = temp.apply(lambda x: a.loc[x["nomprov"]]["renta"] if np.isnan(x["renta"]) and 
                                                                    str(x["nomprov"]) != 'nan' else x["renta"], axis=1 )

d = set(data.columns) - set(["ult_fec_cli_1t" ,"conyuemp","nomprov"])

data = data[list(d)]  
        
#for c in mydata.columns:
    #print(c , data[c].isnull().sum())  
#    print(c, mydata[c].dtypes )

data["segmento"] = data["segmento"].apply(lambda x: 0 if x == "02 - PARTICULARES" else 1 if x == "03 - UNIVERSITARIO" else 2)   

data["ind_empleado"] = data["ind_empleado"].apply(lambda x: 1 if x == "N" else 2 if x == "B" else 3 if x == "F" else 4 if x == "A"
                                                                else 5 )
data["tiprel_1mes"] = data["tiprel_1mes"].apply(lambda x:1 if x == "I" else 2 if x == "A" else 3 if x == "P" else 4 if x == "R"
                                                                else 5 )
data["ind_empleado"] = data["ind_empleado"].apply(lambda x: 1 if x == "N" else 2 if x == "B" else 3 if x == "F" else 4 if x == "A"
                                                                else 5 )

data["sexo"] = data["sexo"].apply(lambda x: 0 if x == "V" else 1)

data["ind_nuevo"] = data["ind_nuevo"].apply(lambda x: 0 if x == '0' else 1)

data["indrel"] = data["indrel"].apply(lambda x: 1 if x == '0' else 0)



data["indresi"] = data["indresi"].apply(lambda x: 1 if x == "S" else 2)

data["indext"] = data["indext"].apply(lambda x: 1 if x == "S" else 2)

data["indfall"] = data["indfall"].apply(lambda x: 1 if x == "S" else 2)

data["tipodom"] = data["tipodom"].apply(lambda x: 1 if x == "1" else 0)

data["ind_actividad_cliente"] = data["ind_actividad_cliente"].apply(lambda x:0 if x == "0" else 1)



data["cod_prov"] = data["cod_prov"].apply(lambda x: int(x))

data["ind_nomina_ult1"] = data["ind_nomina_ult1"].apply(lambda x: 1 if x == 1.0 else 0)

data["ind_nom_pens_ult1"] = data["ind_nom_pens_ult1"].apply(lambda x: 1 if x == 1.0 else 0)

data["indrel_1mes"] = data["indrel_1mes"].apply(lambda x:2 if x == 2 else 3 if x == 3 else 4 if x == "P" else 1)

data["cod_prov"] = data["cod_prov"].astype(int)
data["cod_prov"].fillna(28 , inplace=True)
data["canal_entrada"].fillna("KHE" ,inplace = True)
data["pais_residencia"] = data["pais_residencia"].apply(lambda x: mapd["pais_residencia"][x])

data["pais_residencia"] = data["pais_residencia"].apply(lambda x: mapd["pais_residencia"][x])
data["canal_entrada"] = data["canal_entrada"].apply(lambda x: mapd["canal_entrada"][x])

d = set(mydata.columns) - set(["fecha_alta"])

mydata = mydata[list(d)]
          
mydata.to_csv('test_edited.csv',header=True,cols=mydata.columns)

del data
# Working with cleaned file

products = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
            'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
            'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
             'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

def runXGB(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.05
	param['max_depth'] = 8
	param['silent'] = 1
	param['num_class'] = 24
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.7
	param['colsample_bytree'] = 0.7
	param['seed'] = seed_val
	num_rounds = 50

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model
 
customer = {}    
        
def processtrain(infile , customer):
    x = []
    y = []

    while(1):
        line = infile.readline()[:-1]

        if line == '':
            break
        
        arr = line.split(",")
        
        for i in range(46):
            if i not in [another["fecha_dato"] ,another["renta"]]:
                arr[i] = int(arr[i])
                
        arr[another["renta"]] = float(arr[another["renta"]])  
        
        customerid = int(arr[another["ncodpers"]])
        arr[another["ncodpers"]] = 0
        arr[another['Unnamed: 0']] = 0
        
        prolist = [] 
        another['ind_deme_fin_ult1']
        flag = 1
        
        if arr[another["fecha_dato"]] in ['2015-05-28', '2016-05-28']:	
            arr[another["fecha_dato"]] = 0
            for value in products:
                prolist.append(arr[another[value]])
                customer[customerid] =  prolist[:]
            flag=0
        
        if(flag):
            xtemp = []
            arr[another["fecha_dato"]] = 0
            for value in usersinfo:
                xtemp.append(arr[another[value]])
        
            pre = customer.get(customerid , [0]*24)
            prolist = []
            for value in products:
                prolist.append(arr[another[value]])
                
            new_prods = []
            for i in range(len(pre)):
                new_prods.append(max(prolist[i] - pre[i],0))

            if sum(new_prods) > 0:
                for ind, prod in enumerate(new_prods):
                    if prod>0:
                        assert len(pre) == 24
                        x.append(xtemp+pre)
                        y.append(ind)
   
                
    return x, y, customer         
    
def processtest(infile , customer):
    x = []
    y = []

    infile = test
    while(1):
        line = infile.readline()[:-1]

        if line == '':
            break
        
        arr = line.split(",")
        
        for i in range(21):
            if i not in [another["fecha_dato"],another["renta"]]:
                arr[i] = int(arr[i])
      
        arr[another["renta"]] = float(arr[another["renta"]])
        
        customerid = int(arr[another["ncodpers"]])
        
        arr[another["ncodpers"]] = 0
        xtemp = []

        for value in usersinfo:
            if value == 'fecha_dato':
                xtemp.append(0)
            else:    
                xtemp.append(arr[another[value]])
        
        if arr[another["fecha_dato"]] == '2016-06-28':
            arr[another["fecha_dato"]] = 0
            pre = customer.get(customerid , [0]*24)
            x.append(xtemp + pre)
                
    return x, y, customer         
    
              
train = open("D:/Studies/Online Competitions/Santner recommendation/edited.csv")

line = train.readline()[:-1]

names = line.split(",")
names
another ={}
for i in range(1,46):
    another[names[i]] = i

usersinfo = set(names) - set(products)-set(['' ,'Unnamed: 0'])
usersinfo
x ,y , customer = processtrain(train , {})
x[0]
train_input = np.array(x)
train_output = np.array(y)

del x, y

train.close()

test = open("D:/Studies/Online Competitions/Santner recommendation/test_edited.csv")

line = test.readline()[:-1]

names = line.split(",")
another = {}

for i in range(1,22):
    another[names[i]] = i

usersinfo = set(names) - set(products)-set([''])

x ,y , customer = processtest(test , {})

test_input = np.array(x)
del x
test.close()

model = runXGB(train_input, train_output, seed_val=0)
del train_input, train_output

xgtest = xgb.DMatrix(test_input)

preds = model.predict(xgtest)
del test_input, xgtest

products = np.array(products)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv("D:/Studies/Online Competitions/Santner recommendation/test_edited.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(products[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('sub_xgb_new.csv', index=False)


















































