# -*- coding: utf-8 -*-
"""

******************
*** INCOMPLETE ***
******************

This reads in a text file containing the worked hours and generates an array of
randomly generated numbers for the digital punch card ZEB.

@author: Hristo Goumnerov
"""
#%% Import Modules

# import numpy for advanced mathematical operations
import numpy as np

# import tkinter for file selection dialog
import tkinter
tkinter.Tk().withdraw()
from tkinter.filedialog import askopenfilename as ask

# import prettytable for table display
from prettytable import PrettyTable

#%% Input columns

#n = np.int32(input('Number of columns: '))
#
#c = np.zeros(n,dtype=float)
#
#for i in range(n):
#    c[i] = np.float(input('Day %d: ' % (i+1)))

# Currently working
def gethours(x):
    
    l = len(x)
    
    td = 0          
    
    hours = []
    
    spec1 = ['ZA','FT','UL','KR']
    ids = []
    specs = []
    
    for i in range(l):
        if '001' in x[i]:
            td += 1
            hst = x[i].find('001')
            hours.append(x[i][hst+9:hst+13])
            
        for j in range(len(spec1)):
            if spec1[j] in x[i][:2]:
                ids.append(td)
                specs.append(spec1[j])
                hours.pop()
    
    n = len(hours)
    c = np.zeros(n,dtype=float)

    for k in range(n):
        c[k] = int(hours[k][:1]) + int(hours[k][2:])/60
    
    return n,c,ids,specs

#%% Input projects

def getprojs():
    
    # Number of projects
    m = np.int32(input('Number of rows: '))
    
    # Project hours
    r = np.zeros(m, dtype=float)
    for i in range(m):
        r_in = input('Row %d: ' % (i+1))
        r_in = r_in.replace(',','.')
        r[i] = float(r_in)
        
    return m,r

#%% Output c

# Random matrix generator with inputs: row dimension, column dimension, input 
# row sum, input column sum, max column sum difference

def zebbi(m,n,r,c,cr):
    
    # Loop criterion
    check = 1
    it = 1
    
    # Loop
    while check == 1:
        
        # Initialize matrix
        out_1 = np.zeros((m,n),dtype=np.float)
        
        # Fill matrix with column sums equal to worked hours
        for j in range(n):
            out_1[:,j] = np.random.dirichlet(np.ones(m),size=1)*c[j]
        
        # Round matrix to one decimal place
        out_1 = np.around(out_1, decimals=1)
        
        # Compute row and column sums
        rs = out_1.sum(axis=1)
        
        ###############
        ### Check 1 ###
        ###############
        
        # Compute difference between input and output row sums
        rdiff = r-rs
        
        # Initialize corrected matrix
        out_2 = np.zeros((m,n),dtype=np.float)
        
        # Add or subtract values (average over number of columns) to fit row
        # sums
        for i in range(m):
            for j in range(n):
                if rdiff[i] < 0:
                    out_2[i,j] = out_1[i,j] - abs(rdiff[i])/n
                else:
                    out_2[i,j] = out_1[i,j] + rdiff[i]/n
        
        # Remove negative values by subtracting averaged value from other
        # elements
        for i in range(m):
            for j in range(n):
                if out_2[i,j] < 0:
                    out_2[:,j] = out_2[:,j] - abs(out_2[i,j])/(len(r)-1)
                    out_2[i,j] = 0
        
        # Final output rounded to one decimal place
        output = abs(np.around(out_2, decimals=1))
        
        ###############
        ### Check 2 ###
        ###############
        
        # Compute difference between input and output column sums
        rs = output.sum(axis=1)
        cs = output.sum(axis=0)
        
        ch = 0
        for i in range(m):
            ch += abs(r[i]-rs[i])
        
        ch = ch/m
        print(it)
        it += 1
        
        # Loop condition: 0 < column sum difference < 5 and no negative entries
        if ch < cr and np.all(output >= 0.0):
            check = 0
    
    return output, rs, cs
            
#%% Print table

# Print ZEB matrix with inputs: row dimension, column dimension, matrix
def printzeb(m,n,days,matrix,ids,specs,rin,cin,rs,cs):
    
    matrix = np.array(matrix, dtype=str)
    
    for i in range(len(ids)):
        inc = np.array([specs[i] for j in range(m)], dtype=str)
        idc = ids[i]-1
        if i == 0:
            mat = np.column_stack((matrix[:,:idc],inc,matrix[:,idc:]))
            if idc == 0:
                mat = np.column_stack((inc,matrix[:,idc:]))
        else:
            mat = np.column_stack((mat[:,:idc],inc,mat[:,idc:]))
            if idc == 0:
                mat = np.column_stack((inc,mat[:,idc:]))
       
    # Table output 
    outt = np.array(np.zeros((m+1,len(days)+1)), dtype=str)
    header = days #[str(' Day ' + str((i+1))) for i in range(n)]
    header.insert(0,'Project #')
    outt[0,:] = header
    for i in range(m):
        outt[i+1,0] = 'Project ' + str(i+1) + ':'
        outt[i+1,1:] = mat[i,:]
    
    t = PrettyTable()
    t.field_names = outt[0,:]
    for i in range(1,outt.shape[0]):
      t.add_row(outt[i,:])
    
    c = cin
    r = rin
    
    print(t)
    print('\nSum of INPUT columns: %4.1f' % c.sum())
    print('Sum of OUTPUT columns: %4.1f' % cs.sum())
    print('Sum of INPUT rows: %4.1f' % r.sum())
    print('Sum of OUTPUT rows: %4.1f' % rs.sum())
    
    return outt

#%% Run code

# Filename
fn = ask() #'okt.txt'

# Read file
with open(fn, 'r') as file:
    data = [line for line in file.readlines()]

# Process file
days = []
dc = 0
for i in range(len(data)):
    if data[i].startswith('---'):
        dc = i

days = data[:dc]
for i in range(dc):
    if days[i][2] == '.':
        days[i] = days[i][:5]
    else:
        days[i] = '0'

days = [x for x in days if x!='0' and x[3:]!='Sa' and x[3:]!='So']

n,c,ids,specs = gethours(data)
m,r = getprojs()
out,rs,cs = zebbi(m,n,r,c,1)
outt = printzeb(m,n,days,out,ids,specs,r,c,rs,cs)
