"""Shared own-code object core (no zaeval import). Per tile, per model: 8-conn components of the binary mask with
n_px, n_valid, valid-px intersections with gt_id; border pixel->comp map (std cross-tile merge); and for the 0.25 m
unit metric, the label of each comp in the 8-conn labelling of the mask dilated by a Euclidean disk r=2.5 px (0.125 m),
plus that dilated labelling's border map (cross-tile dmerge)."""
import numpy as np
from scipy import ndimage
S8=np.ones((3,3),bool)
yy,xx=np.mgrid[-2:3,-2:3]; DISK=(yy**2+xx**2)<=6.25
def hyst(p,hi,lo):
    lab,nl=ndimage.label(p>=lo,S8)
    if nl==0: return np.zeros(p.shape,bool)
    return np.r_[False,ndimage.maximum(p,lab,np.arange(1,nl+1))>=hi][lab]
def border(H,W):
    ys=np.r_[np.zeros(W,int),np.full(W,H-1),np.arange(1,H-1),np.arange(1,H-1)]; xs=np.r_[np.arange(W),np.arange(W),np.zeros(H-2,int),np.full(H-2,W-1)]; return ys,xs
def tile_parts(pm,valid,gid,gx0,gy0):
    H,W=pm.shape; ys,xs=border(H,W)
    lab,nl=ndimage.label(pm,S8)
    if nl==0: return None
    lr=lab.ravel(); npx=np.bincount(lr,minlength=nl+1); nva=np.bincount(lr[valid.ravel()],minlength=nl+1)
    bo=valid.ravel()&(lr>0)&(gid.ravel()>0); inter=[dict() for _ in range(nl+1)]
    if bo.any():
        pr,cnt=np.unique(np.stack([lr[bo],gid.ravel()[bo]]),axis=1,return_counts=True)
        for (c,o),n in zip(pr.T,cnt): inter[c][int(o)]=int(n)
    dl,_=ndimage.label(ndimage.binary_dilation(pm,DISK),S8)
    cd=ndimage.maximum(dl,lab,np.arange(1,nl+1)).astype(int)  # comp -> dilated label (comp pixels are inside its dilated blob)
    comps=[(int(npx[c]),int(nva[c]),inter[c],int(cd[c-1])) for c in range(1,nl+1)]
    bl=lab[ys,xs]; f=bl>0; e={(gx0+int(x),gy0+int(y)):int(c)-1 for y,x,c in zip(ys[f],xs[f],bl[f])}
    db=dl[ys,xs]; f=db>0; de={(gx0+int(x),gy0+int(y)):int(c) for y,x,c in zip(ys[f],xs[f],db[f])}
    return dict(comps=comps,edges=e,dedges=de)
