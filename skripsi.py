import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA
import scipy.stats as stats
from pymongo import MongoClient
import json

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Choose your database and collection
db = client['db-untar-cafe']
students_collection = db['members']
nonCoding_collection = db['noncodingstudents']

# Fetch data from both collections
coding_students = list(students_collection.find({}, {"grade": 1, "science": 1, "tech": 1, "math": 1, "_id": 0}))
non_coding_students = list(nonCoding_collection.find({}, {"grade": 1, "science": 1, "tech": 1, "math": 1, "_id": 0}))

# Convert to Pandas DataFrame if needed for calculations
coding_df = pd.DataFrame(coding_students)
non_coding_df = pd.DataFrame(non_coding_students)

coding_df['group'] = 'Coding'
non_coding_df['group'] = 'NonCoding'

df = pd.concat([coding_df, non_coding_df], ignore_index=True)

# Take data by subject
dataAsliScience = df['science']
dataAsliTech = df['tech']
dataAsliMath = df['math']


# Grandmean each subject
grandMeanAsliS = dataAsliScience.mean()
grandMeanAsliT = dataAsliTech.mean()
grandMeanAsliM = dataAsliMath.mean()

# centering data
centeringDataS = dataAsliScience - grandMeanAsliS
centeringDataT = dataAsliTech - grandMeanAsliT
centeringDataM = dataAsliMath - grandMeanAsliM

# Standard deviasi
stdScience = dataAsliScience.std()
stdTech = dataAsliTech.std()
stdMath = dataAsliMath.std()

# Scaling data
scalingDataS = centeringDataS / stdScience
scalingDataT = centeringDataT / stdTech
scalingDataM = centeringDataM / stdMath

dataNewS = pd.concat([df['group'], df['grade'], scalingDataS], axis=1)
dataNewT = pd.concat([df['group'], df['grade'], scalingDataT], axis=1)
dataNewM = pd.concat([df['group'], df['grade'], scalingDataM], axis=1)


# Data coding after scaling
codingS = dataNewS[dataNewS['group'] == 'Coding']
codingT = dataNewT[dataNewT['group'] == 'Coding']
codingM = dataNewM[dataNewM['group'] == 'Coding']

#  Data noncoding after scaling
nonCodingS = dataNewS[dataNewS['group'] == 'NonCoding']
nonCodingT = dataNewT[dataNewT['group'] == 'NonCoding']
nonCodingM = dataNewM[dataNewM['group'] == 'NonCoding']

# Grandmean after scaling
grandMeanS = (round(codingS['science'].mean(),2) + round(nonCodingS['science'].mean(),2)) / 2
grandMeanT = (round(codingT['tech'].mean(),2) + round(nonCodingT['tech'].mean(),2)) / 2
grandMeanM = (round(codingM['math'].mean(),2) + round(nonCodingM['math'].mean(),2)) / 2

grandMean = [grandMeanS, grandMeanT, grandMeanM]
grandMean = np.matrix(grandMean)

I = 2 # total group (coding dan non coding)
J = 2 # total group (kelas 1-3 dan kelas 4-6)
K = len(dataAsliScience) # total data
d = 3 # total variable independent
mE = I*J*(K - 1) # degrees of freedom error
v1 = d
v2 = mE + 1 - d

# find y for coding group
ySC = codingS['science'].mean()
yTC = codingT['tech'].mean()
yMC = codingM['math'].mean()

yiCoding = [ySC, yTC, yMC]
yiCoding = np.matrix(yiCoding)

y1 = yiCoding - grandMean
y1T = y1.T

SSA1 = J * np.dot(y1T,y1)

#  find y for noncoding group
ySnC = nonCodingS['science'].mean()
yTnC = nonCodingT['tech'].mean()
yMnC = nonCodingM['math'].mean()

yinonCoding = [ySnC, yTnC, yMnC]
yinonCoding = np.array(yinonCoding)

y2 = yinonCoding - grandMean
y2T = y2.T

SSA2 = J * np.dot(y2T,y2)

# Factor A
factorA = np.round(SSA1 + SSA2, decimals=6)

# Take data for grade 1-3 coding
dataScience13 = codingS[codingS['grade'].isin([1,2,3])]
dataTech13 = codingT[codingT['grade'].isin([1,2,3])]
dataMath13 = codingM[codingM['grade'].isin([1,2,3])]

# drop column group and grade
dataScience13 = dataScience13.drop(['group', 'grade'], axis=1)
dataTech13 = dataTech13.drop(['group', 'grade'], axis=1)
dataMath13 = dataMath13.drop(['group', 'grade'], axis=1)

yijcS13 = dataScience13.mean()
yijcT13 = dataTech13.mean()
yijcM13 = dataMath13.mean()
yijCoding13 = np.matrix([yijcS13,yijcT13,yijcM13])

# Take data for grade 1-3 non coding
nondataScience13 = nonCodingS[nonCodingS['grade'].isin([1,2,3])]
nondataTech13 = nonCodingT[nonCodingT['grade'].isin([1,2,3])]
nondataMath13 = nonCodingM[nonCodingM['grade'].isin([1,2,3])]

# drop column group and grade
nondataScience13 = nondataScience13.drop(['group', 'grade'], axis=1)
nondataTech13 = nondataTech13.drop(['group', 'grade'], axis=1)
nondataMath13 = nondataMath13.drop(['group', 'grade'], axis=1)

yijnS13 = nondataScience13.mean()
yijnT13 = nondataTech13.mean()
yijnM13 = nondataMath13.mean()
yijnonCoding13 = np.matrix([yijnS13,yijnT13,yijnM13])

# Take data for grade 4-6 coding
dataScience46 = codingS[codingS['grade'].isin([4,5,6])]
dataTech46 = codingT[codingT['grade'].isin([4,5,6])]
dataMath46 = codingM[codingM['grade'].isin([4,5,6])]

# drop column group and grade
dataScience46 = dataScience46.drop(['group', 'grade'], axis=1)
dataTech46 = dataTech46.drop(['group', 'grade'], axis=1)
dataMath46 = dataMath46.drop(['group', 'grade'], axis=1)

yijcS46 = dataScience46.mean()
yijcT46 = dataTech46.mean()
yijcM46 = dataMath46.mean()
yijCoding46 = np.matrix([yijcS46,yijcT46,yijcM46])

# Take data for grade 4-6 noncoding
nondataScience46 = nonCodingS[nonCodingS['grade'].isin([4,5,6])]
nondataTech46 = nonCodingT[nonCodingT['grade'].isin([4,5,6])]
nondataMath46 = nonCodingM[nonCodingM['grade'].isin([4,5,6])]

# drop column group and grade
nondataScience46 = nondataScience46.drop(['group', 'grade'], axis=1)
nondataTech46 = nondataTech46.drop(['group', 'grade'], axis=1)
nondataMath46 = nondataMath46.drop(['group', 'grade'], axis=1)

yijnS46 = nondataScience46.mean()
yijnT46 = nondataTech46.mean()
yijnM46 = nondataMath46.mean()
yijnonCoding46 = np.matrix([yijnS46,yijnT46,yijnM46])

# Mean for each subject
meanyjS13 = np.concatenate([np.ravel(dataScience13), np.ravel(nondataScience13)])
meanyjT13 = np.concatenate([np.ravel(dataTech13), np.ravel(nondataTech13)])
meanyjM13 = np.concatenate([np.ravel(dataMath13), np.ravel(nondataMath13)])
yjS13 = meanyjS13.mean()
yjT13 = meanyjT13.mean()
yjM13 = meanyjM13.mean()
yj13 = np.matrix([yjS13, yjT13, yjM13])

meanyjS46 = np.concatenate([np.ravel(dataScience46), np.ravel(nondataScience46)])
meanyjT46 = np.concatenate([np.ravel(dataTech46), np.ravel(nondataTech46)])
meanyjM46 = np.concatenate([np.ravel(dataMath46), np.ravel(nondataMath46)])
yjS46 = meanyjS46.mean()
yjT46 = meanyjT46.mean()
yjM46 = meanyjM46.mean()
yj46 = np.matrix([yjS46, yjT46, yjM46])

yj1 = yj13 - grandMean
SSB1 = I * np.dot(yj1.T, yj1)

yj2 = yj46 - grandMean
SSB2 = I * np.dot(yj2.T, yj2)

factorB = np.round(SSB1 + SSB2, decimals=6)


#Factor AB
h31 = yijCoding13 - yiCoding - yj13 + grandMean
H31 = np.dot(h31.T, h31)

h32 = yijnonCoding13 - yinonCoding - yj13 + grandMean
H32 = np.dot(h32.T, h32)

h33 = yijCoding46 - yiCoding - yj46 + grandMean
H33 = np.dot(h33.T, h33)

h34 = yijnonCoding46 - yinonCoding - yj46 + grandMean
H34 = np.dot(h34.T, h34)

factorAB = H31 + H32 + H33 + H34

dataCoding13 = pd.concat([dataScience13, dataTech13, dataMath13], axis=1)
dataCoding46 = pd.concat([dataScience46, dataTech46, dataMath46], axis=1)
nondataCoding13 = pd.concat([nondataScience13, nondataTech13, nondataMath13], axis=1)
nondataCoding46 = pd.concat([nondataScience46, nondataTech46, nondataMath46], axis=1)

yijCoding13 = np.array(yijCoding13.T)
yijCoding46 = np.array(yijCoding46.T)
yijnonCoding13 = np.array(yijnonCoding13.T)
yijnonCoding46 = np.array(yijnonCoding46.T)

errordataCoding13 = dataCoding13 - yijCoding13

errordataCoding46 = dataCoding46 - yijCoding46

errornondataCoding13 = nondataCoding13 - yijnonCoding13

errornondataCoding46 = nondataCoding46 - yijnonCoding46

errorCoding13 = []
errordataCoding13 = np.matrix(errordataCoding13)
for i in range(errordataCoding13.shape[0]):
    result = np.dot(errordataCoding13[i].T, errordataCoding13[i])
    errorCoding13.append(result)   
    
hasilCoding13 = sum(errorCoding13)

errorCoding46 = []
errordataCoding46 = np.matrix(errordataCoding46)
for i in range(errordataCoding46.shape[0]):
    result = np.dot(errordataCoding46[i].T, errordataCoding46[i])
    errorCoding46.append(result)   
    
hasilCoding46 = sum(errorCoding46)

errornonCoding13 = []
errornondataCoding13 = np.matrix(errornondataCoding13)
for i in range(errornondataCoding13.shape[0]):
    result = np.dot(errornondataCoding13[i].T, errornondataCoding13[i])
    errornonCoding13.append(result)   
    
hasilnonCoding13 = sum(errornonCoding13)

errornonCoding46 = []
errornondataCoding46 = np.matrix(errornondataCoding46)
for i in range(errornondataCoding46.shape[0]):
    result = np.dot(errornondataCoding46[i].T, errornondataCoding46[i])
    errornonCoding46.append(result)   
    
hasilnonCoding46 = sum(errornonCoding46)

error = hasilCoding13 + hasilCoding46 + hasilnonCoding13 + hasilnonCoding46

he1 = error + factorA
he2 = error + factorB
he3 = error + factorAB
lambda1 = np.linalg.det(error) / np.linalg.det(he1)
lambda2 = np.round(np.linalg.det(error) / np.linalg.det(he2), decimals=6)
lambda3 = np.round(np.linalg.det(error) / np.linalg.det(he3), decimals=6)
# print('Lambda1' , lambda1)


F0h1 = np.round(((1-lambda1)/lambda1) * ((mE + 1 - d)/d), decimals= 4)
m1 = I - 1 # degrees of freedom factor A
# print(F0h1)
Ftabel = np.round(stats.f.ppf(0.95, v1, v2), decimals=4)

    
F0h2 = np.round(((1-lambda2)/lambda2) * ((mE + 1 - d)/d), decimals=4)
m2 = J - 1 # degrees of freedom factor B
v1 = d
v2 = mE + 1 - d


F0h3 = np.round(((1-lambda3)/lambda3) * ((mE + 1 - d)/d), decimals=4)
m3 = (I - 1)/ (J - 1) # degrees of freedom factor AxB

if F0h1 > Ftabel:
    h_group = "Hasil H0 ditolak, terdapat perbedaan signifikan antara siswa Coding dan Non-Coding."
else:
    h_group = "Hasil H0 diterima, tidak terdapat perbedaan signifikan antara siswa Coding dan Non-Coding."

if F0h2 > Ftabel:
    h_grade = "Hasil H0 ditolak, terdapat perbedaan signifikan antara siswa kelas 1-3 dan kelas 4-6."
else:
    h_grade = "Hasil H0 diterima, tidak terdapat perbedaan signifikan antara siswa kelas 1-3 dan kelas 4-6."

if F0h3 > Ftabel:
    h_h3 = "Hasil H0 ditolak, terdapat perbedaan signifikan antara siswa Coding dan Non-Coding pada kelas 1-3 dan kelas 4-6 ."
else:
    h_h3 = "Hasil H0 diterima, tidak ada perbedaan signifikan antara siswa Coding dan Non-Coding pada kelas 1-3 dan kelas 4-6."

v1 = d
v2 = mE + 1 - d


data = pd.DataFrame({
    'Science': df['science'],
    'Technology': df['tech'],
    'Math': df['math'],
    'Group': df['group'],  # Coding/Non-Coding
    'Grade': df['grade']    # 1-3 or 4-6
})


EH1 =np.linalg.inv(error + factorA) 
pillai1 = np.round(np.trace(factorA @ EH1), decimals=4)

EH2 =np.linalg.inv(error + factorB) 
pillai2 = np.round(np.trace(factorB @ EH2), decimals=4)

EH3 =np.linalg.inv(error + factorAB) 
pillai3 = np.round(np.trace(factorAB @ EH3), decimals=4)

alpha = 0.05
if pillai1 < alpha:
    uji_group = "Berdasarkan uji statistik Pillai H0 ditolak, terdapat perbedaan signifikan antara siswa Coding dan Non-Coding."
else:
    uji_group = "Berdasarkan uji statistik Pillai H0 diterima, tidak terdapat perbedaan signifikan antara siswa Coding dan Non-Coding."

if pillai2 < alpha:
    uji_grade = "Berdasarkan uji statistik Pillai H0 ditolak, terdapat perbedaan signifikan antara siswa kelas 1-3 dan kelas 4-6."
else:
    uji_grade = "Berdasarkan uji statistik Pillai H0 diterima, tidak terdapat perbedaan signifikan antara siswa kelas 1-3 dan kelas 4-6."

if pillai3 < alpha:
    uji_h3 = "Berdasarkan uji statistik Pillai H0 ditolak, terdapat perbedaan signifikan antara siswa Coding dan Non-Coding pada kelas 1-3 dan kelas 4-6 ."
else:
    uji_h3 = "Berdasarkan uji statistik Pillai H0 diterima, tidak ada perbedaan signifikan antara siswa Coding dan Non-Coding pada kelas 1-3 dan kelas 4-6."
    
# Hasil perhitungan disimpan dalam dictionary
hasil_analisis = {
    'Ftabel': Ftabel,
    'F0h1': F0h1,
    'F0h2': F0h2,
    'F0h3': F0h3,
    'h_group': h_group,
    'h_grade': h_grade,
    'h_h3': h_h3,
    'Pillai1': pillai1,
    'Pillai2': pillai2,
    'Pillai3': pillai3,
    'uji_group': uji_group,
    'uji_grade': uji_grade,
    'uji_h3': uji_h3,
    
}

# Simpan hasil ke file JSON
with open('hasil_analisis.json', 'w') as json_file:
    json.dump(hasil_analisis, json_file)