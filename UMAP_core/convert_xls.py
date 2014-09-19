import numpy as np
import xlwt

#load .txt data
data = np.loadtxt("/Users/071cht/Desktop/Lab/lab_data/diagnosiscandidate_matrix.txt",delimiter= ',') 
flyID = np.genfromtxt("/Users/071cht/Desktop/Lab/lab_data/dlmcell/diaflyID.txt") 
"""I haven't figure out how to put originalID on it yet"""

#hard code the number of colomns here 748/4=187
#We have to seperate data in to three matrix because python xlwt can only write excel with the 256 col at most.
data1=data[:,0:248] #62 flies
data2=data[:,248:496] #62 flies
data3=data[:,496:748] #63 flies

#set up workbook
book = xlwt.Workbook()
sheet1 = book.add_sheet('dandidates1', cell_overwrite_ok=True)
sheet2 = book.add_sheet('dandidates2', cell_overwrite_ok=True)
sheet3 = book.add_sheet('dandidates3', cell_overwrite_ok=True)

for i in range(len(data1)):
    for j in range(len(data1[i])):
        sheet1.write(i+2,j,data1[i][j])
        if data1[i][j]==99:
        	 sheet1.row(i+2).set_cell_blank(j)

for i in range(len(data2)):
    for j in range(len(data2[i])):
        sheet2.write(i+2,j,data2[i][j])
        if data2[i][j]==99:
        	 sheet2.row(i+2).set_cell_blank(j)

for i in range(len(data3)):
    for j in range(len(data3[i])):
        sheet3.write(i+2,j,data3[i][j]) 
        if data3[i][j]==99:
        	 sheet3.row(i+2).set_cell_blank(j)  

#Add header
labels= np.array(['frame','Brad_predict','JAABA_predict','True'])
idx_sheet1= np.arange(0,data1.shape[1],4)
for j in idx_sheet1:
	sheet1.row(1).write (j,labels[0])
	sheet1.row(1).write (j+1,labels[1])
	sheet1.row(1).write (j+2,labels[2])
	sheet1.row(1).write (j+3,labels[3])
	sheet1.write_merge(0,0,j,j+3,'target_number:'+' '+str(flyID[j/4][1].astype(int)))


idx_sheet2= np.arange(0,data2.shape[1],4)
idx_sheet2= np.arange(0,data2.shape[1],4)
for j in idx_sheet2:
	sheet2.row(1).write (j,labels[0])
	sheet2.row(1).write (j+1,labels[1])
	sheet2.row(1).write (j+2,labels[2])
	sheet2.row(1).write (j+3,labels[3])
	sheet2.write_merge(0,0,j,j+3,'target_number:'+' '+str(flyID[62+j/4][1].astype(int)))


idx_sheet3= np.arange(0,data3.shape[1],4)
idx_sheet3= np.arange(0,data3.shape[1],4)
for j in idx_sheet3:
	sheet3.row(1).write (j,labels[0])
	sheet3.row(1).write (j+1,labels[1])
	sheet3.row(1).write (j+2,labels[2])
	sheet3.row(1).write (j+3,labels[3])
	sheet3.write_merge(0,0,j,j+3,'target_number:'+' '+str(flyID[124+j/4][1].astype(int)))    	 

name = "/Users/071cht/Desktop/Lab/lab_data/abc.xls"
book.save(name)
