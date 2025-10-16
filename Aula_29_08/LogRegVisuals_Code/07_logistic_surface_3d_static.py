import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
def sigmoid(z): return 1/(1+np.exp(-z))
def main(out="logistic_surface_3d.png"):
    data=load_breast_cancer(); X=data.data; y=data.target; names=data.feature_names
    i1=list(names).index("mean radius"); i2=list(names).index("mean texture")
    X=X[:,[i1,i2]]; X=StandardScaler().fit_transform(X)
    w=np.zeros(2); b=0.0; lr=0.1
    for _ in range(200):
        z=X@w+b; yhat=sigmoid(z); w-=lr*(X.T@(yhat-y))/len(y); b-=lr*np.mean(yhat-y)
    x1=np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,120); x2=np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,120)
    X1,X2=np.meshgrid(x1,x2); Z=sigmoid(w[0]*X1+w[1]*X2+b)
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
    ax.plot_surface(X1,X2,Z,linewidth=0,antialiased=True,alpha=0.9)
    ax.contour(X1,X2,Z,levels=[0.5],zdir='z',offset=0,linestyles='--')
    ax.set_xlabel(f"{names[i1]} (std)"); ax.set_ylabel(f"{names[i2]} (std)"); ax.set_zlabel("P(y=1|x)")
    ax.set_zlim(0,1); ax.set_xlim(x1.min(),x1.max()); ax.set_ylim(x2.min(),x2.max())
    ax.view_init(elev=25, azim=-60); plt.tight_layout(); plt.savefig(out,dpi=120,bbox_inches="tight")
if __name__=="__main__": main()
