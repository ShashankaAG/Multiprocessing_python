# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:18:56 2022

@author: SHASHU
"""

from multiprocessing import Pool
from pytictoc import TicToc
import os
import numpy as np
import csv

tt=TicToc()

dt          = 1#int(input('Enter the step size = '))                        # seconds
tspan       = 2*24*60*60
R_E         = 6378.0 # km
earth_mu    = 3.9860043543609598E+14 # m^3 / s^2

def order2_ode( t, state):
    r = state[ :3 ]
    a = -earth_mu *r/np.linalg.norm(r) ** 3

    return np.array( [ state[ 3 ], state[ 4 ], state[ 5 ],
             a[ 0 ], a[ 1 ], a[ 2 ] ] )

def rk4( f, t, y, h ):
    k1 = f( t, y )
    k2 = f( t + 0.5 * h, y + 0.5 * k1 * h )
    k3 = f( t + 0.5 * h, y + 0.5 * k2 * h )
    k4 = f( t + h, y + k3 * h )

    return y + h / 6.0 * ( k1 + 2 * k2 + 2 * k3 + k4 )


steps       = int( tspan / dt )
ets         = np.zeros( ( steps, 1 ) )
states      = np.zeros( ( steps, 6 ) )
time_orb    = np.zeros( ( steps, 1 ) )

def runkaro(yo):  
    
    states[ 0 ] = yo
    time_orb[0] = 0
    row_counter=0
    for step in range( steps - 1 ):
        states[ step + 1 ] = rk4(
        order2_ode, ets[ step ], states[ step ], dt )
        time_orb[step +1] = time_orb[step]+dt
            
    return time_orb,states[:,0:3]

workers = 1#int(input('Enter the number of workers = '))#os.cpu_count()
print('No. of workers:',workers,'& step size :',dt)
tt.tic()
if __name__ == '__main__':
    with Pool(workers) as p:
        if workers==1:
            A = p.map(runkaro, [[-297805.1775,6744160.841,425915.8493,
                                  -5562.321025,100.2956944,-5289.177796]])
        elif workers==2:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]])
        elif workers==3:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0]])
        elif workers==4:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0],[4509e3,0,4509e3,0,7.7e3,0]])
        elif workers==5:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0],[4509e3,0,4509e3,0,7.7e3,0]
                                       ,[4509e3,0,4509e3,0,7.6e3,0]])
        elif workers==6:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0],[4509e3,0,4509e3,0,7.7e3,0]
                                       ,[4509e3,0,4509e3,0,7.6e3,0],[4509e3,0,4509e3,0,7.5e3,0]])
        elif workers==7:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0],[4509e3,0,4509e3,0,7.7e3,0]
                                       ,[4509e3,0,4509e3,0,7.6e3,0],[4509e3,0,4509e3,0,7.5e3,0]
                                       ,[4509e3,0,4509e3,0,7.4e3,0]])
        elif workers==8:
            A = p.map(runkaro, [[4509e3,0,4509e3,0,8e3,0],[4509e3,0,4509e3,0,7.9e3,0]
                                       ,[4509e3,0,4509e3,0,7.8e3,0],[4509e3,0,4509e3,0,7.7e3,0]
                                       ,[4509e3,0,4509e3,0,7.6e3,0],[4509e3,0,4509e3,0,7.5e3,0]
                                       ,[4509e3,0,4509e3,0,7.4e3,0],[4509e3,0,4509e3,0,7.3e3,0]])
        p.close()
        p.join()
tt.toc()

#%% writing data to csv using csv package and numpy package(commented out)
for i in range(len(A)):
    lpl = 'dt_'+str(dt)+'_'+'wrks_'+str(workers)+'_'+str(i+1)+'.csv'
    B=np.concatenate([A[i][0].flat,A[i][1][:,0].flat,A[i][1][:,1].flat,A[i][1][:,2].flat])
    B=B.reshape(4,len(A[0][1])).T
    # np.savetxt(lpl, B,delimiter=',',fmt='%d')  #numpy
    data_list =[["time", "X", "Y","Z"]]
    with open(lpl, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data_list)
        writer.writerows(B)