from re import X
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1.0,1.0,101)
y = np.abs(x)+0.3*x

plt.plot(x,np.abs(x),"--")
plt.plot(x,np.abs(y))
plt.savefig("abs.svg")
plt.savefig("abs.png")

plt.clf()


fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(x,x*x,"--")
ax.plot(x,x*x*(x+1))
ax.set_xlim(-1,0.75)
ax.set_ylim(-0.1,0.3)
plt.savefig("cubic.svg")
plt.savefig("cubic.png")

x = np.linspace(-5.0,2.0,701)

fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(x,x*x,"--")
ax.plot(x,x*x*(x+1),":")
ax.plot(x,x*x*np.exp(x))
ax.set_xlim(-1,0.75)
ax.set_ylim(-0.1,1.0)

plt.savefig("exp1.svg")
plt.savefig("exp1.png")

fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(x,x*x,"--")
ax.plot(x,x*x*(x+1),":")
ax.plot(x,x*x*np.exp(x))
ax.set_xlim(-5,2)
ax.set_ylim(-0.1,1.0)

plt.savefig("exp2.svg")
plt.savefig("exp2.png")

def sigmoid(x,k=1):
    return 1/(1+np.exp(-k*x))

x = np.linspace(-10.0,10.0,200)
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(x,sigmoid(x,0.5),label="k=1/2")
ax.plot(x,sigmoid(x,1),label="k=1")
ax.plot(x,sigmoid(x,2),label="k=2")
ax.legend()
ax.set_xlim(-10,10)
ax.set_ylim(-0.1,1.1)

plt.savefig("sigmoid.svg")
plt.savefig("sigmoid.png")

plt.show()
