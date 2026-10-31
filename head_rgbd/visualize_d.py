"""CPU z-buffer geometry views with shared lighting, no texture."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from head_rgbd.fit_b import normals


def render(vertices, faces, center, yaw, smooth=True, size=360):
    angle=np.radians(yaw)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    q=(vertices-center)@rotation.T
    vn=normals(q,faces)
    triangles=q[faces]
    fn=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-15)
    uv=np.column_stack((q[:,0]*size/.36+size/2,-q[:,1]*size/.36+size/2))
    depth=np.full((size,size),-np.inf); rgb=np.full((size,size,3),.94)
    light=np.array([-.35,.45,1.]);light/=np.linalg.norm(light)
    for i,face in enumerate(faces):
        t=uv[face]; x0,y0=np.maximum(np.ceil(t.min(0)).astype(int),0)
        x1,y1=np.minimum(np.floor(t.max(0)).astype(int),[size-1,size-1])
        if x1<x0 or y1<y0:continue
        x,y=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
        a,b,c=t; den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        b0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        b1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        bc=np.stack((b0,b1,1-b0-b1),axis=-1); z=bc@q[face,2]
        patch=depth[y0:y1+1,x0:x1+1]; inside=(bc>=-1e-8).all(-1)&(z>patch)
        n=bc@vn[face] if smooth else np.broadcast_to(fn[i],bc.shape).copy()
        n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12)
        shade=.21+.74*np.maximum(n@light,0)
        color=np.repeat(shade[:,:,None],3,axis=2)
        patch[inside]=z[inside];rgb[y0:y1+1,x0:x1+1][inside]=color[inside]
    return rgb


def comparison_views(methods, root, center):
    for smooth in (False,True):
        fig,axes=plt.subplots(3,len(methods),figsize=(3*len(methods),8.4),squeeze=False)
        for col,(name,v,f) in enumerate(methods):
            for row,yaw in enumerate((0,30,90)):
                axes[row,col].imshow(render(v,f,center,yaw,smooth))
                axes[row,col].set_title(f'{name}\nview {yaw:+d} deg',fontsize=9)
                axes[row,col].axis('off')
        fig.suptitle('Shared light and scale | '+('interpolated vertex normals' if smooth else 'face normals')+' | 90 deg is a mesh view, not an observation',fontsize=11)
        fig.tight_layout()
        fig.savefig(root/f'geometry_{"smooth" if smooth else "flat"}.png',dpi=130)
        plt.close(fig)
