import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("samochody.csv")

X = df[['Waga', 'Objetosc']]
y = df['CO2']

skala_x = scale.fit_transform(X)

regresja = linear_model.LinearRegression()
regresja.fit(skala_x, y)

#Waga pojazdu | objetosc silnika 
scaled = scale.transform([[2500, 1.3]])

przewidywalnyCO2 = regresja.predict([scaled[0]])
print(przewidywalnyCO2)