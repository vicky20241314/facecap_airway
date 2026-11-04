"""Comparable physical-scale diagnostics and explicit triangle risk screening."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import dijkstra,connected_components
import trimesh
from rtree import index
from head_rgbd.fit_b import normals
from head_rgbd.geometry_d import child_landmarks


def summary(values):
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def topology_summary(faces):
    directed=np.vstack((faces[:,:2],faces[:,1:],faces[:,[2,0]]))
    edges,inverse,count=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    signs=np.bincount(inverse,weights=np.where(directed[:,0]<directed[:,1],1.,-1.))
    boundary=edges[count==1];nodes,compressed=np.unique(boundary,return_inverse=True)
    components=0
    if len(nodes):
        b=compressed.reshape(-1,2)
        graph=sparse.coo_matrix((np.ones(len(b)),(b[:,0],b[:,1])),shape=(len(nodes),len(nodes))).tocsr()
        components=connected_components(graph+graph.T,directed=False,return_labels=False)
    return dict(unique_edges=len(edges),boundary_edges=len(boundary),boundary_vertices=len(nodes),
        boundary_components=int(components),nonmanifold_edges=int((count>2).sum()),
        inconsistent_two_face_edge_winding=int(((count==2)&(signs!=0)).sum()),
        boundary_branch_or_endpoint_vertices=int((np.bincount(compressed,minlength=len(nodes))!=2).sum()),
        watertight_topology=bool((count==2).all()),boundary_vertex_indices=nodes)


def physical_samples(vertices, faces, coarse_faces, dense=False, *, subsamples=1):
    parent = np.arange(len(coarse_faces))
    bc = np.full((len(parent), 3), 1/3)
    if subsamples==4:
        # All four child interiors, including the three corner triangles. The
        # exact same barycentric material coordinates are used on coarse meshes.
        parent=np.repeat(parent,4)
        bc=np.tile(np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3],[1/3,1/3,1/3]]),(len(coarse_faces),1))
    elif subsamples!=1:raise ValueError('supported material quadrature: 1 or 4')
    fi, bc = child_landmarks(parent, bc) if dense else (parent, bc)
    triangles = vertices[faces[fi]]
    point = (triangles*bc[:, :, None]).sum(1)
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-15)
    return point, normal


def physical_roughness(vertices, faces, base_vertices, coarse_faces, supported_faces, dense=False, radius_m=.006, *, subsamples=1, return_samples=False):
    # Same material points and same 6mm neighborhoods at EVERY resolution.
    point, normal = physical_samples(vertices, faces, coarse_faces, dense,subsamples=subsamples)
    base_point, _ = physical_samples(base_vertices, coarse_faces, coarse_faces)
    adjacency=trimesh.Trimesh(base_vertices,coarse_faces,process=False).face_adjacency
    lengths=np.linalg.norm(base_point[adjacency[:,0]]-base_point[adjacency[:,1]],axis=1)
    graph=sparse.coo_matrix((np.r_[lengths,lengths],
        (np.r_[adjacency[:,0],adjacency[:,1]],np.r_[adjacency[:,1],adjacency[:,0]])),shape=(len(coarse_faces),len(coarse_faces))).tocsr()
    selected=np.flatnonzero(supported_faces)
    neighborhoods={}
    for start in range(0,len(selected),64):
        indices=selected[start:start+64]
        distance=dijkstra(graph,directed=False,indices=indices,limit=radius_m)
        for i,row in zip(indices,distance):neighborhoods[int(i)]=np.flatnonzero(np.isfinite(row))
    angles, plane_rms, sample_faces = [], [], []
    t=base_vertices[coarse_faces]
    reference_area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    for i in np.flatnonzero(supported_faces):
        ids = np.asarray(neighborhoods[i], int)
        ids = ids[supported_faces[ids]]
        if len(ids) < 6:
            continue
        weight=np.repeat(reference_area[ids]/subsamples,subsamples)
        ids=(ids[:,None]*subsamples+np.arange(subsamples)).ravel()
        weight/=max(weight.sum(),1e-20)
        n = (normal[ids]*weight[:,None]).sum(0)
        n /= max(np.linalg.norm(n), 1e-12)
        angles.append(np.sqrt(np.sum(weight*np.degrees(np.arccos(np.clip(normal[ids]@n, -1, 1)))**2)))
        q = point[ids]-(point[ids]*weight[:,None]).sum(0)
        _, _, v = np.linalg.svd(q*np.sqrt(weight[:,None]), full_matrices=False)
        plane_rms.append(np.sqrt(np.sum(weight*(q@v[-1])**2))*1000)
        sample_faces.append(i)
    result=dict(radius_mm=radius_m*1000, metric_version='geodesic_material_area_quadrature_v3',subsamples_per_parent=subsamples,
                sampling='same material quadrature in every resolution; original area weights; fixed base 6mm dual-graph paths; excludes disconnected layers',
                normal_spread_rms_deg=summary(angles), local_best_plane_rms_mm=summary(plane_rms))
    if return_samples:
        result['samples']=dict(parent_faces=np.asarray(sample_faces,dtype=int),normal_spread_rms_deg=np.asarray(angles),
                               local_best_plane_rms_mm=np.asarray(plane_rms))
    return result


def coplanar_overlap_area(first,second):
    """2D convex clipping. Positive area includes identical triangles; contact is zero."""
    def orient(a,b,c):return np.cross(b-a,c-a)
    sign=1. if orient(second[0],second[1],second[2])>=0 else -1.
    polygon=[p.copy() for p in first]
    for i in range(3):
        if not polygon:return 0.
        a,b=second[i],second[(i+1)%3];result=[]
        previous=polygon[-1];dp=float(sign*orient(a,b,previous))
        for current in polygon:
            dc=float(sign*orient(a,b,current))
            if (dc>=0)!=(dp>=0):result.append(previous+(current-previous)*(dp/(dp-dc)))
            if dc>=0:result.append(current)
            previous,dp=current,dc
        polygon=result
    if len(polygon)<3:return 0.
    polygon=np.asarray(polygon)
    return float(abs(np.cross(polygon,np.roll(polygon,-1,axis=0)).sum())*.5)


def segment_triangle(start, end, triangle):
    direction = end-start
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    e1, e2 = b-a, c-a
    h = np.cross(direction, e2)
    determinant = (e1*h).sum(1)
    valid = np.abs(determinant) > 1e-15
    inverse = np.zeros_like(determinant); inverse[valid] = 1/determinant[valid]
    s = start-a
    u = inverse*(s*h).sum(1)
    q = np.cross(s, e1)
    v = inverse*(direction*q).sum(1)
    t = inverse*(e2*q).sum(1)
    return valid&(u >= -1e-8)&(v >= -1e-8)&(u+v <= 1+1e-8)&(t > 1e-8)&(t < 1-1e-8)


def triangle_risks(vertices, faces, *, include_pair_ids=False):
    vertices=np.asarray(vertices,dtype=np.float64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    options = index.Property(); options.dimension = 3
    lo, hi = triangles.min(1)-1e-10, triangles.max(1)+1e-10
    tree = index.Index(((i, tuple(np.r_[a,b]), None) for i,(a,b) in enumerate(zip(lo,hi))), properties=options)
    pairs = []
    for i in range(len(faces)):
        candidate = np.array([j for j in tree.intersection(tuple(np.r_[lo[i],hi[i]])) if j > i], int)
        if len(candidate):
            shared = (faces[candidate,:,None]==faces[i][None,None,:]).any((1,2))
            pairs.extend((i,int(j)) for j in candidate[~shared])
    tree.close()
    hits = np.zeros(len(pairs), bool)
    coplanar_hits = np.zeros(len(pairs), bool)
    if pairs:
        ids = np.array(pairs)
        first, second = triangles[ids[:,0]], triangles[ids[:,1]]
        for edge in range(3):
            hits |= segment_triangle(first[:,edge], first[:,(edge+1)%3], second)
            hits |= segment_triangle(second[:,edge], second[:,(edge+1)%3], first)
        n1, n2 = cross[ids[:,0]], cross[ids[:,1]]
        n1 /= np.maximum(np.linalg.norm(n1,axis=1,keepdims=True),1e-15)
        n2 /= np.maximum(np.linalg.norm(n2,axis=1,keepdims=True),1e-15)
        coplanar = (np.linalg.norm(np.cross(n1,n2),axis=1)<1e-7)&(np.abs(((second[:,0]-first[:,0])*n1).sum(1))<1e-9)
        for k in np.flatnonzero(coplanar):
            axis = int(np.argmax(np.abs(n1[k])))
            a, b = np.delete(first[k],axis,axis=1), np.delete(second[k],axis,axis=1)
            coplanar_hits[k] = coplanar_overlap_area(a,b)>1e-14
    result=dict(finite=bool(np.isfinite(vertices).all()), vertices=len(vertices), triangles=len(faces),
                degenerate_double_area_below_1e_14=int((double_area<1e-14).sum()),
                minimum_double_area_m2=float(double_area.min()),
                nonincident_aabb_pairs=len(pairs), proper_intersection_pairs=int(hits.sum()),
                coplanar_overlap_pairs=int(coplanar_hits.sum()),
                intersection_pair_examples=[pairs[i] for i in np.flatnonzero(hits|coplanar_hits)[:25]],
                self_intersection_scope='nonincident triangle segment crossings and coplanar area overlap; incident overlaps/tangencies not certified')
    if include_pair_ids:
        result['proper_intersection_pair_ids']=[pairs[i] for i in np.flatnonzero(hits)]
        result['coplanar_overlap_pair_ids']=[pairs[i] for i in np.flatnonzero(coplanar_hits)]
    return result
