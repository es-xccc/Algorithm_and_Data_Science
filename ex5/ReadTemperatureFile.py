# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:54:38 2023

@author: tiw
"""
def getTemperatureData(fileName, cities): 
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
    print(cities,'cities total temperature:\n', TotalYearTemperature)            
    dataFile.close() 
    return TotalYearTemperature 
