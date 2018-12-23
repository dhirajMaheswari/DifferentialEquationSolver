'''
Created on May 4, 2018

@author: Dhiraj
this is the code to solve first order differential equation of the form
dy/dx = g(x,y), with boundary condition known such that y(x = x0) = y0
solution is y(x)

 the exact is  y(x) = exp(-cos(x))/(cosh(1)-sinh(1)) for g(x,y) = sin(x)*y
'''
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import matplotlib.animation as anim

def g(x,y):
#    return mt.sin(x)*y
     return 6.0*y**2*x


def FiniteDiff(x0,y0,xn,h,g):
    '''x0 = initial x coordinate, y0 = intial condition [value of y(x0)
       xn = final x coordinate, h = step size
       g-> rhs of the differential equation dy/dx = g(x,y) 
       for first order Diff Eqn
    '''
    n = (xn - x0)/h # number of points to estimate the finite differences at
    x = np.linspace(x0,xn,n) # obtain the x coordinates in the range
    #print("Number of points: ",n, " length of x: ", len(x))
    y = np.zeros(len(x))
    y[0] = y0 # assign the initial value 
    xcord = []
    soln = []
    for j in range(0,len(y)-1):
        y[j+1] = y[j] + h*g(x[j],y[j])
       # print(x[j], y[j+1])
        xcord.append(x[j])
        soln.append(y[j+1])
    return xcord,soln   # return the lists containing xcoordinates and the corresponding solution 

x,y = FiniteDiff(1,1.0/25.0,3.0, 0.05, g)
exactSoln = []
for e in x:
    #exactSoln.append(mt.exp(-mt.cos(e))/(mt.cosh(1) - mt.sinh(1)))
    exactSoln.append(1.0/(28.0 - 3.0*e**2))


print("Number of points: ", len(y))

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution of first order Differential Equation")
plt.scatter(x, y,c = 'red', marker= '*')
#plt.scatter(x,y, c = 'red', market = '*', label = "calculated")
ax = plt.gca()
plt.plot(x,exactSoln, c = 'blue')
#plt.plot(x,exactsoln, label = 'exact soln')
ax.legend(('calculated', 'exact solution'))
#plt.legend(loc = 'best')

plt.show()


def g2(x,y):
    return mt.sin(x)

def FiniteDiff2ndOrder(x0,y0, yprime0, xn, h, g2):
    '''x0 = initial x coordinate, y0 = intial condition [value of y(x0)
        yprime0 : y'(x0) , initial value to be provided. 
       xn = final x coordinate, h = step size
       g-> rhs of the differential equation d^2y/dx^2 = g(x,y) 
       for first order Diff Eqn
    '''
    n = (xn - x0)/h # number of points to estimate the finite differences at
    x = np.linspace(x0,xn,n) # obtain the x coordinates in the range
    #print("Number of points: ",n, " length of x: ", len(x))
    y = np.zeros(len(x))
    y[0] = y0 # assign the initial value 
    xcord = []
    soln = []
    for j in range(0,len(x)-1):
        y[j+1] = 2.0*h*yprime0 + (1.0-h**2)*g2(x[j],y[j])
        #print(x[j], y[j+1])
        xcord.append(x[j])
        soln.append(y[j+1])
    return xcord,soln   # return the lists containing xcoordinates and the corresponding solution 


x,y = FiniteDiff2ndOrder(0,1,0,2.0*mt.pi,0.05,g2)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solution of second order Differential Equation")
plt.plot(x, y,c = 'red', marker= '*')
#plt.scatter(x,y, c = 'red', market = '*', label = "calculated")
ax = plt.gca()
#plt.plot(x,exactsoln, label = 'exact soln')
ax.legend(('calculated'))
#plt.legend(loc = 'best')

plt.show()


    