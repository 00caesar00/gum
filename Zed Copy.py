from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from varname import nameof
import sys
import time as tm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.formatter.useoffset'] = False
#We aim to simulate encounters of a single H- ion (negative hydrogen ion) and  
#a H+ (proton), where the H+ is trapped in an ideal Penning trap and the H- is the projectile.
#The center of the Penning trap is the origin and the magnetic field is along the z-#axis.
#The H+ is initially at the origin or oscillating close to the origin, the H- is injected #from a point on or close to the negative z-axis, with a velocity in the positive
#z-direction, so that it interacts with the H+ with low energy (~10 meV) in the
#center of mass (that is, it is injected with just enough energy to nearly reach, or just #overcome the potential hill that is the axial trap for the H+ ion.)
#By running the code with different initial conditions for the ions, we aim to
#characterize the transfer of energy between the respective axial, cyclotron and
#magnetron motions of the ions.

print('program starting now')

# initialization of variables
B = 5 #Magnetic field (T) [Not Used]
epsilon0 = 8.854e-12 #Vacuum Permittivity (F/m)
mp = 1.672e-27  #Mass of proton (kg)
me = 9.109e-31 #Mass of electron (kg)
e = 1.602e-19   #Charge of proton (C)

#We specify the Penning trap by specifying the cyclotron frequency of the  proton
#instead of the B-field, and the axial frequency of the proton, instead of the trap #voltage and trap size parameter.

om = 2*np.pi*1e8 #6.28e8
omz = 2*np.pi*1e6
#Ion-1 is the proton, ion-2 is the H-
m1, m2=  mp, mp+2*me  
omc1, omc2 = om, (m1/m2)*om
omz1 = omz
omz2 = omz1*np.sqrt(m1/m2)

#Since we need to specify accelerations, we specify the strength of the Coulomb
#interaction between the ions with parameters divided by ion mass.
c1,c2 = 0,0 #e**2/(4*np.pi*epsilon0*m1),e**2/(4*np.pi*epsilon0*m2)
N = 500000 # Number of steps [om*tStep << 2*pi] 
tStep = float(input("Desired time step in Seconds: "))#(tMax - tMin)/N  # time step 
tMin, tMax = 0.0, N*tStep #1e-7  # time range
#Number of steps
#Desired Step size
time = np.arange(tMin,tMax,tStep)
Coulomb = str(input("Calculate Coulomb interaction? Enter 'y' for yes 'n' for no: "))

def C(arg):
  return (np.cos(arg) - 1)

def S(arg):
  return (np.sin(arg) - arg)

def a(x1, y1, z1, x2, y2, z2):    # Calculates accelerations given positions of both particles
  r = np.sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))
  if Coulomb.lower()=="y":    # Checking to see if simulation should consider coulomb interaction or not
    ax1 =  (omz1**2)*x1/2 +(c1/(r)**3)*(x2-x1)
    ay1 =  (omz1**2)*y1/2 +(c1/(r)**3)*(y2-y1)
    az1 = (omz1**2)*z1 +(c1/(r)**3)*(z2-z1)
    
 
    ax2 =  -(omz2**2)*x2/2 +(c2/(r)**3)*(x1-x2)
    ay2 =  -(omz2**2)*y2/2 +(c2/(r)**3)*(y1-y2)
    az2 =  (omz2**2)*z2 +(c1/(r)**3)*(z1-z2) ################This was changed from (omz2**2)* *********1/9/21 sign on omz**2*z changed to switch field polarity on ALL az**********
  if Coulomb.lower()=="n":
    ax1 =  (omz1**2)*x1/2
    ay1 =  (omz1**2)*y1/2
    az1 = -(omz1**2)*z1##-
 
    ax2 =  -(omz2**2)*x2/2 
    ay2 =  -(omz2**2)*y2/2 
    az2 =  (omz2**2)*z2 ################This was changed from (omz2**2)*    
  
  a_product = np.array([ax1, ay1, az1, ax2, ay2, az2], float) #Returns array with components of each acceleration

  return a_product

def xcalc(vx, vy, ax, ay, rx): # Calculates the x component of the position
   return (rx + (1/om)*(vx*np.sin(om*tStep)-vy*C(om*tStep))+(1/om**2)*((-ax)*C(om*tStep)-ay*S(om*tStep)))

def ycalc(vx, vy, ax, ay, ry): # Calculates the y component of the position
   return (ry - (1/om)*(vy*np.sin(-om*tStep)-vx*C(-om*tStep))+(1/om**2)*((-ay)*C(-om*tStep)-ax*S(-om*tStep)))

def zcalc(rz,vz,az):           # Calculates the z component of the position
  return(rz + tStep*vz + (1/2)*(tStep**2)*az)

def vxcalc(vx, vy, ax, ay, ax_Step, ay_Step): # Calculates the x component of the velocity
  return vx*np.cos(om*tStep)+vy*np.sin(om*tStep)+(1/om)*((-ay)*C(om*tStep)+ax*np.sin(om*tStep))-(1/om**2)*(((ax_Step-ax)/tStep)*C(om*tStep)+((ay_Step-ay)/tStep)*S(om*tStep))

def vycalc(vx, vy, ax, ay, ax_Step, ay_Step): # Calculates the y component of the velocity
  return vy*np.cos(-om*tStep)+vx*np.sin(-om*tStep)-(1/om)*((-ax)*C(-om*tStep)+ay*np.sin(-om*tStep))-(1/om**2)*(((ay_Step-ay)/tStep)*C(-om*tStep)+((ax_Step-ax)/tStep)*S(-om*tStep))

def vzcalc(vz,az,az_Step):                    # Calculates the z component of the velocity
    return vz + tStep*(az+az_Step)/2


def calculation(vec): #Function runs V-V algorithm and returns Position/Velocity/Acceleration in an array for time t+dt
    #vec format is (rx1, vx1, ax1, ry1, vy1, ay1, rz1, vz1, az1, rx2, vx2, ax2, ry2, vy2, ay2, rz2, vz2, az2)
    #First calculate next positions of each particle
    x1_Step = xcalc(vec[1], vec[4], vec[2], vec[5], vec[0])
    y1_Step = ycalc(vec[1], vec[4], vec[2], vec[5], vec[3])
    z1_Step = zcalc(vec[6], vec[7], vec[8])
    Position1 = [x1_Step,y1_Step,z1_Step]
   
   
    x2_Step = xcalc(vec[10], vec[13], vec[11], vec[14], vec[9])
    y2_Step = ycalc(vec[10], vec[13], vec[11], vec[14], vec[12])
    z2_Step = zcalc(vec[15], vec[16], vec[17])
    Position2 = [x2_Step,y2_Step,z2_Step]

    #Secondly, calculate the next accelerations of each particle
    a1_Step = a(x1_Step,y1_Step,z1_Step,x2_Step,y2_Step,z2_Step)[:3] #[:3] takes only the first three entries of array "a"
    a2_Step = a(x1_Step,y1_Step,z1_Step,x2_Step,y2_Step,z2_Step)[3:] #[3:] takes final three entries of array "a"
    Acceleration1 = a1_Step
    Acceleration2 = a2_Step
   
    #Lastly, calculate the next velocities of each particle
    #vec format is (rx1, vx1, ax1, ry1, vy1, ay1, rz1, vz1, az1, rx2, vx2, ax2, ry2, vy2, ay2, rz2, vz2, az2)
    vx1_Step = vxcalc(vec[1], vec[4], vec[2], vec[5], a1_Step[0], a1_Step[1])
    vy1_Step = vycalc(vec[1], vec[4], vec[2], vec[5], a1_Step[0], a1_Step[1])
    vz1_Step = vzcalc(vec[7], vec[8], a1_Step[2])
    Velocity1 = [vx1_Step,vy1_Step,vz1_Step]
    vx2_Step = vxcalc(vec[10], vec[13], vec[11], vec[14], a2_Step[0], a2_Step[1])
    vy2_Step = vycalc(vec[10], vec[13], vec[11], vec[14], a2_Step[0], a2_Step[1])
    vz2_Step = vzcalc(vec[16], vec[17], a2_Step[2])
    Velocity2 = [vx2_Step,vy2_Step,vz2_Step]

    return Position1, Position2, Acceleration1, Acceleration2, Velocity1, Velocity2

def Plot_Focus(time1,time2,Y_Value_Array,Time_Array,Coordinate_Number,Title,Y_Label):
  time1/=tStep
  time2/=tStep
  time1,time2 = int(time1), int(time2)
  plt.scatter(Time_Array[time1:time2],Y_Value_Array[time1:time2,Coordinate_Number],s=.5)
  plt.title(Title)
  plt.xlabel('Time, Seconds')
  plt.ylabel(Y_Label)
  plt.savefig('Last_Run'+str(Coordinate_Number)) 
  plt.xlim(Time_Array[time1],Time_Array[time2])
  plt.show()
  plt.close() 

def main():
   #Initialize arrays to store position/velocity/acceleration data
   #Arrays are of size N (Number of calculation steps) and contain sub arrays of size 3
    Particle1_Position, Particle2_Position = np.zeros((N, 3), float), np.zeros((N,3), float)
   
    Particle1_Velocity, Particle2_Velocity = np.zeros((N,3), float), np.zeros((N,3), float)
   
    Particle1_Acceleration, Particle2_Acceleration = np.zeros((N,3), float), np.zeros((N,3), float)
    #Initial Positions
    xi1,yi1,zi1 = 0.00, 0.01, -0.01 
    xi2,yi2,zi2 = 0.0, 0.00, .0
    #Initial Velocities
    vxi1,vyi1,vzi1 = -0.01*om,0, (omz*0.01)*0.999999 
    vxi2,vyi2,vzi2 = 0.0, 0.0,0.00
    #Calculating initial acceleration from initial conditions
    a0 = a(xi1, yi1, zi1, xi2, yi2, zi2) #Accelerations (rx10,ry10,rz10,rx20,ry20,rz20)
    #Updating vec with proper initial accelerations
    vec = np.array([xi1, vxi1, a0[0], yi1, vyi1, a0[1], zi1, vzi1, a0[2], xi2, vxi2, a0[3], yi2, vyi2, a0[4], zi2, vzi2, a0[5]], float)
    #vec format is (rx1, vx1, ax1, ry1, vy1, ay1, rz1, vz1, az1, rx2, vx2, ax2, ry2, vy2, ay2, rz2, vz2, az2)
    for t in range(N): #Iterating through each time step, updating with calculation function and storing data in Position, Velocity, and Acceleration arrays
        ''' 
        print("[rx10, vx10, ax10, ry10")
        print("vy10, ay10, rz10, vz10")
        print("az10, rx20, vx20, ax20")
        print("ry20, vy20, ay20, rz20")
        print("vz20, az20]")
        print(vec)
        input("Continue?")
        '''
        Particle1_Position[t], Particle2_Position[t], Particle1_Acceleration[t], Particle2_Acceleration[t], Particle1_Velocity[t], Particle2_Velocity[t] = calculation(vec)
        
        
        vec = np.array([Particle1_Position[t , 0], Particle1_Velocity[t , 0], Particle1_Acceleration[t , 0],  #X coordinates
                        Particle1_Position[t , 1], Particle1_Velocity[t , 1], Particle1_Acceleration[t , 1],  #Y coordinates
                        Particle1_Position[t , 2], Particle1_Velocity[t , 2], Particle1_Acceleration[t , 2],  #Z coordinates
                        Particle2_Position[t , 0], Particle2_Velocity[t , 0], Particle2_Acceleration[t , 0],  #X                                                                                                            
                        Particle2_Position[t , 1], Particle2_Velocity[t , 1], Particle2_Acceleration[t , 1],  #Y
                        Particle2_Position[t , 2], Particle2_Velocity[t , 2], Particle2_Acceleration[t , 2]]) #Z
        if t%(int(N/100))==0:  #Simple manual % Complete readout for longer simulations
            #input("continue?")
            print(str(int((t/N)*100))+chr(37)+" complete")
            sys.stdout.write("\033[F")
            
    
    print("The time step was", tStep,"seconds")

### X position of orbit first brief time
    fig=plt.figure()
    plt.scatter(time,Particle1_Position[:,0],s=.5)
    Title = 'X-Position vs. Time'
    Y_Label = 'X Position for Proton, Meters'
    plt.title(Title)
    plt.xlabel('Time, Seconds')
    plt.ylabel(Y_Label)
    plt.savefig('Radius') 
    plt.xlim(tMin,time[5000])
    plt.show()
    plt.close()

  
### Radius of orbit
    fig=plt.figure()
    plt.scatter(time,np.sqrt(np.add(np.square(Particle1_Position[:,0]),np.square(Particle1_Position[:,1]))),s=.5)
    plt.title('Radius of orbit vs. Time')
    plt.xlabel('Time, Seconds')
    plt.ylabel('Orbit Radius for Proton, Meters')
    plt.savefig('Radius')
    plt.xlim(tMin,tMax)
    plt.show()
    plt.close()

### Y Position
    plt.scatter(time,Particle1_Position[:,1],s=.5)
    Title = 'Y-position for Proton vs Time'
    Y_Label = 'Y-position for Proton, Meters'
    plt.title(Title)
    plt.xlabel('Time, Seconds')
    plt.ylabel(Y_Label)
    plt.savefig('ymotion1')
    plt.xlim(tMin,tMax)
    plt.show()
    plt.close()

    #Plot_Focus(0,0.00000005,Particle1_Position,time,1,Title,Y_Label)


    #Z Position Graph
    plt.scatter(time,Particle1_Position[:,2],s=.5)
    plt.title('Z-position Proton vs. Time')
    plt.xlabel('Time, Seconds')
    plt.ylabel('Z-pos for Proton, Meters')
    plt.savefig('zmotion1')
    plt.xlim(tMin,tMax)
    plt.show()
    plt.close()


    #Energy Graphs (eV)
    #X-Energy (Proton)
    plt.scatter(time, 0.5*m1*(1/e)*(Particle1_Velocity[:,0])**2, s=.5) # Z - Component Energy
    plt.title('X-Direction Kinetic Energy (eV) vs Time (s)')
    plt.show()
    plt.close()

    #Y-Energy (Proton)
    plt.scatter(time, 0.5*m1*(1/e)*(Particle1_Velocity[:,1])**2, s=.5) # Z - Component Energy
    plt.title('Y-Direction Kinetic Energy (eV) vs Time (s)')
    plt.show()
    plt.close()
    #Z-Energy (Proton)
    plt.scatter(time, 0.5*m1*(1/e)*(Particle1_Velocity[:,2])**2, s=.5) # Z - Component Energy
    plt.title('Z-Direction Kinetic Energy (eV) vs Time (s)')
    plt.show()
    plt.close()

'''
### 3D Figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(Particle1_Position[:,0], Particle1_Position[:,1], Particle1_Position[:,2], zdir ='z')
    ax.plot(Particle2_Position[:,0], Particle2_Position[:,1], Particle2_Position[:,2], zdir='z')
    ax.text2D(0.05, 0.95, "Proton Position State Space", transform=ax.transAxes)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.savefig('totalmotion.png')
    plt.show()
    plt.close()
'''

main()