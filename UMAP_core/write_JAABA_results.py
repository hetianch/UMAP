import numpy as np
import xlwt

path='/Users/071cht/Desktop/for_conference_longer/test3/exp/'
data = np.loadtxt(path+'JAABA_result_matrix.txt',delimiter= ',') 

fin = open(path+'originalID.txt', "r")
flyID = fin.readlines()
fin.close()

flyID = [item.rstrip('\r\n') for item in flyID]

if data.shape[0] % 65535 ==0:
	sheetnumber=(data.shape[0]/65535 ).astype(int)
else:
	sheetnumber = (np.floor(data.shape[0]/65535 + 1)).astype(int)

#print data.shape[1],sheetnumber
data_sepindex=np.zeros([sheetnumber]).astype(int)

for i in range (0,sheetnumber):
	data_sepindex[i]= i*65535

#set up workbook

sheet_array= np.empty([sheetnumber])
book = xlwt.Workbook()

#sheet_name_list= []
#for i in range (0,sheetnumber):
#	sheet_name_list.append(str(i+1))
#print type (sheet_name_list),type(sheet_name_list[1])

# handle the first to the second last sheet
labels= np.array(['flyID','frame','multifly'])

for sheet in range (0,sheetnumber-1):
	sheet_index=str(sheet)
	sheet_to_add= book.add_sheet(sheet_index, cell_overwrite_ok=True)
	sheet_to_add.row(0).write (0,labels[0])
	sheet_to_add.row(0).write (1,labels[1])
	sheet_to_add.row(0).write (2,labels[2]) 

	for row in range(0,65535):
	    sheet_to_add.write(row+1,1,data[data_sepindex[sheet]+row][0])
	    sheet_to_add.write(row+1,2,data[data_sepindex[sheet]+row][1])
	    sheet_to_add.write(row+1,0,flyID[data_sepindex[sheet]+row])  




# handle the last sheet
sheet_to_add= book.add_sheet(str(sheetnumber-1), cell_overwrite_ok=True)
row_start=data.shape[1]-data_sepindex[sheetnumber-1]
sheet_to_add.row(0).write (0,labels[0])
sheet_to_add.row(0).write (1,labels[1])
sheet_to_add.row(0).write (2,labels[2])
for row in range(0,len(data)-data_sepindex[sheetnumber-1]):
	sheet_to_add.write(row+1,1,data[data_sepindex[sheetnumber-1]+row][0])
	sheet_to_add.write(row+1,2,data[data_sepindex[sheetnumber-1]+row][1])
	sheet_to_add.write(row+1,0,flyID[data_sepindex[sheetnumber-1]+row]) 
	  
	


name = path+'JAABAresults_spreadsheet.xls'
book.save(name)
