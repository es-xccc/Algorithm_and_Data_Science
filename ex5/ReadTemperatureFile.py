# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:54:38 2023

@author: tiw
"""
import matplotlib.pyplot as plt
def getTemperatureData(cities, fileName = "TemperatureofThreecities.txt"): 
    dataFile = open(fileName, 'r') 
    TotalYearTemperature = []
    for i in range(cities):
        city=dataFile.readline()
        cityTotalYearTemperature=[city[:-2]]
#        print(city)        
        for year in range(10):
#            print(year+2013)
            cityYearTemperature=[]
            line=dataFile.readline()[:-1] 
#            print(line)
            yearTemperature = line.split(',')
            tempData=[int(yearTemperature[0])]
            for term in yearTemperature[1:]:
                tempData.append(float(term))
            cityYearTemperature=tempData            
#            print(cityYearTemperature)
            cityTotalYearTemperature.append(cityYearTemperature)
#        print(city,cityTotalYearTemperature)
        TotalYearTemperature.append(cityTotalYearTemperature)
#    print(cities,'cities total temperature:\n', TotalYearTemperature)            
    dataFile.close() 
    return TotalYearTemperature 

city_name = ["Tainan", "Taipei", "Kaohsiung"]
data = getTemperatureData(3)
itr = 0
for city_data in data:
    plt.figure(itr)
    x = list(range(1, 13))
    for year_data in city_data[1:]:
        label = year_data[0]
        y_values = year_data[1:]
        plt.plot(x ,y_values, label=str(label))
    plt.xticks(range(1, 13))
    plt.legend(loc='best', prop={'size': 6})
    plt.title(city_name[itr%3] + " Monthly Mean Temperature From 2013 To 2022")
    plt.xlabel('Month')
    plt.ylabel('Temperature in Degree C')
    plt.savefig(city_name[itr%3] + " Monthly Mean Temperature From 2013 To 2022" + ".png")
    itr += 1

for city_data in data:
    plt.figure(itr)
    x = list(range(1, 13))
    mon_avg_temp = []
    for i in range(12):
        year_sum = 0
        for j in range(10):
            year_sum += city_data[j+1][i+1]
        mon_avg_temp.append(round(year_sum/10, 2))
    allavg = round(sum(mon_avg_temp)/12, 2)
    plt.axhline(y=allavg, color='r', linestyle='--')
    plt.text(1, allavg, str(allavg), fontsize=8)
    plt.plot(x, mon_avg_temp, 'ro')
    plt.plot(x, mon_avg_temp, label="Mean of 10 Years")
    for i in range(len(x)):
        plt.text(x[i], mon_avg_temp[i], str(mon_avg_temp[i]), fontsize=8)
    plt.xticks(range(1, 13))
    plt.legend(loc='best', prop={'size': 6})
    plt.title(city_name[itr%3] + " Monthly Mean Temperature Of 2013 To 2022")
    plt.xlabel('Month')
    plt.ylabel('Temperature in Degree C')
    plt.savefig(city_name[itr%3] + " Monthly Mean Temperature Of 2013 To 2022" + ".png")
    itr += 1