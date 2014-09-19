# this is a file to try out using pbd from command line.
# To open terminal: shift + command + P, then type terminal
# To add current path to your python serach path : 
# >import sys
# >sys.path.append ("current_path")

#Or you can just change the working directory for ipython by :

# import os
# os.getcwd() # get current directory
#os.chdir('current path') # change to current path


#another efficient way to do this is to open a sublimeREPL-python pbd 
#open sublimeREPL-python bd: shift +command + P, then type sublimeREPL and choose from the available consol
import pdb
a = "aaa"
pdb.set_trace()
b = "bbb"
c = "ccc"
final = a + b + c
print final