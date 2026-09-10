"""Deterministic inspection renders, not geometry decisions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS = {"collision":"#f02837","screw":"#aeb7c1","wheel":"#343a46","retainer":"#7f8c99","body":"#abcbd8", "bottom_cover":"#45566c", "battery":"#e5ac4e", "controller":"#47a482", "drive":"#cc6964", "switch":"#be85c6"}


def render_meshes(path, meshes, title, views=((24,-45),(0,0),(90,-90),(0,90)), transparent=False):
    fig = plt.figure(figsize=(14,10), facecolor="#f5f6f8")
    allv = np.concatenate([m.vertices for _,m in meshes if len(m.vertices)])
    lo=allv.min(0);hi=allv.max(0);span=np.maximum(hi-lo,1);margin=span*.06
    prepared=[]
    for name,m in meshes:
        if len(m.faces)>35000:
            import trimesh
            # Native simplification needs writable owned C-contiguous buffers;
            # geometry-library views can be read-only despite correct dtype.
            preview=trimesh.Trimesh(np.array(m.vertices,dtype=np.float64,order='C',copy=True),
                                   np.array(m.faces,dtype=np.int64,order='C',copy=True),process=False)
            m=preview.simplify_quadric_decimation(face_count=35000)
        color=next((c for k,c in COLORS.items() if k in name),"#d3a675")
        light=np.array([.3,-.5,.8]);light/=np.linalg.norm(light)
        shade=.45+.55*np.maximum(0,m.face_normals@light)
        colors=np.column_stack([np.clip(np.array(to_rgb(color))[None,:]*shade[:,None],0,1),np.full(len(m.faces),.18 if transparent and name=="body" else 1.)])
        prepared.append((name,m,colors))
    labels=["X left [mm]","Y front [mm]","Z up [mm]"]
    projections={1:([1,2],0,"Side / YZ"),2:([0,1],2,"Top / XY"),3:([0,2],1,"Front / XZ")}
    for i,(elev,azim) in enumerate(views):
        if i==0:
            ax=fig.add_subplot(2,2,i+1,projection="3d");ax.set_proj_type('ortho')
            for name,m,colors in prepared:ax.add_collection3d(Poly3DCollection(m.triangles,facecolor=colors,edgecolor="none"))
            ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_zlim(lo[2]-margin[2],hi[2]+margin[2])
            ax.set_box_aspect(span);ax.view_init(elev,azim);ax.set_title("Isometric")
            ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.set_zlabel(labels[2])
        else:
            axes,depth,label=projections[i];ax=fig.add_subplot(2,2,i+1)
            triangles=np.concatenate([m.triangles for _,m,_ in prepared]);colors=np.concatenate([c for _,_,c in prepared])
            order=np.argsort(triangles.mean(1)[:,depth],kind='stable')
            ax.add_collection(PolyCollection(triangles[order][:,:,axes],facecolors=colors[order],edgecolors='none'))
            ax.set_xlim(lo[axes[0]]-margin[axes[0]],hi[axes[0]]+margin[axes[0]]);ax.set_ylim(lo[axes[1]]-margin[axes[1]],hi[axes[1]]+margin[axes[1]])
            ax.set_aspect('equal');ax.set_xlabel(labels[axes[0]]);ax.set_ylabel(labels[axes[1]]);ax.set_title(label);ax.grid(alpha=.18)
    fig.suptitle(title,fontsize=16)
    fig.tight_layout(); fig.savefig(path,dpi=125); plt.close(fig)


def candidate_plot(path, outline, groups, title):
    fig,ax = plt.subplots(figsize=(8,11))
    p=outline.vertices
    ax.scatter(p[::max(1,len(p)//7000),0],p[::max(1,len(p)//7000),1],s=.5,c="#b8c2cb")
    for label,positions in groups:
        p=np.asarray(positions)
        if len(p): ax.scatter(p[:,0],p[:,1],s=6,label=label,alpha=.6)
    ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("X [mm]");ax.set_ylabel("Y [mm]");ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
