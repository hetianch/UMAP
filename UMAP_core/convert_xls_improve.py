import numpy as np
import xlwt

#load .txt data
path='/Users/071cht/Desktop/for_conference_longer/test0/output/gentleboost/'
data = np.loadtxt(path+'diagnosiscandidate_matrix.txt',delimiter= ',') 
flyID = np.genfromtxt(path+'flyID.txt') 

"""I haven't figure out how to put originalID on it yet"""



#We have to seperate data in to three matrix because python xlwt can only write excel with the 256 col at most. so we
# seperate data by 248 (#62 flies)colomns per sheet.

if data.shape[1] % 248 ==0:
	sheetnumber=(data.shape[1]/248 ).astype(int)
else:
	sheetnumber = (np.floor(data.shape[1]/248 + 1)).astype(int)

#print data.shape[1],sheetnumber
data_sepindex=np.zeros([sheetnumber]).astype(int)

for i in range (0,sheetnumber):
	data_sepindex[i]= i*248 

#set up workbook

sheet_array= np.empty([sheetnumber])
book = xlwt.Workbook()

#sheet_name_list= []
#for i in range (0,sheetnumber):
#	sheet_name_list.append(str(i+1))
#print type (sheet_name_list),type(sheet_name_list[1])

# handle the first to the second last sheet
labels= np.array(['frame','Brad_predict','multifly probability','True'])
colomn_group= np.arange(0,248,4)

for sheet in range (0,sheetnumber-1):
	sheet_index=str(sheet)
	sheet_to_add= book.add_sheet(sheet_index, cell_overwrite_ok=True)

	for row in range(len(data)):
	    for col in range(0,248):
	        sheet_to_add.write(row+2,col,data[row][data_sepindex[sheet]+col])
	        if data[row][data_sepindex[sheet]+col]<0:
	        	 sheet_to_add.row(row+2).set_cell_blank(col)

	for j in colomn_group:
		sheet_to_add.row(1).write (j,labels[0])
		sheet_to_add.row(1).write (j+1,labels[1])
		sheet_to_add.row(1).write (j+2,labels[2])
		sheet_to_add.row(1).write (j+3,labels[3])
		sheet_to_add.write_merge(0,0,j,j+3,'target_number:'+' '+str(flyID[sheet*62+j/4][1].astype(int)))

# handle the last sheet
labels= np.array(['frame','Brad_predict','multifly probability','True'])
colomn_group= np.arange(0,data.shape[1]-248*(sheetnumber-1),4)
sheet_to_add= book.add_sheet(str(sheetnumber), cell_overwrite_ok=True)

for row in range(len(data)):
	col_number=data.shape[1]-data_sepindex[sheetnumber-1]
	for col in range(0,col_number.astype(int)):
	    sheet_to_add.write(row+2,col,data[row][data_sepindex[sheetnumber-1]+col])
	    if data[row][data_sepindex[sheetnumber-1]+col]<0:
	        sheet_to_add.row(row+2).set_cell_blank(col)

#Add header for the last sheet
for j in colomn_group:
	sheet_to_add.row(1).write (j,labels[0])
	sheet_to_add.row(1).write (j+1,labels[1])
	sheet_to_add.row(1).write (j+2,labels[2])
	sheet_to_add.row(1).write (j+3,labels[3])
	sheet_to_add.write_merge(0,0,j,j+3,'target_number:'+' '+str(flyID[(sheetnumber-1)*62+j/4][1].astype(int)))	 

name = path+'diagnosis_spreadsheet.xls'
book.save(name)
