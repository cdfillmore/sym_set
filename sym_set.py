import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist
import gudhi
import itertools as it
import os
import collections
import colorsys
import igl
from georg_miniball3d import circumsphere_3d
import random as rd

def write_obj(path, verts, simps, name = "test.001"):
    dedup_tris = set([tuple(list(i)) for i in simps])
    dedup_tris = np.array(list(dedup_tris))

    with open(path,'w') as f:
        f.write("# Blender v2.79 (sub 0) OBJ File: '{}.blend'\n# www.blender.org\no {}\n".format(name, name))
        for v in verts:
            f.write('v ' + ' '.join([str(i) for i in v]) + '\n')
        f.write("s off\n")
        for s in simps:
            if len(s) == 2:
                f.write('l ' + ' '.join([str(i+1) for i in s]) + '\n')
            else:
                f.write('f ' + ' '.join([str(i+1) for i in s]) + '\n')
    pass

def read_obj(path):
    pts = []
    simps = []
    with open(path,"r") as f:
        for line in f:
            if line[0] == 'v':
                pts += [[float(i) for i in line.strip('\n').strip(' ').split(' ')[1:4]]]
            elif line[0] == 'f':
                simps += [[ int(i)-1 for i in line.strip('\n').strip(' ').split(' ')[1:4]]]
    return np.array(pts), np.array(simps)

def read_obj_normals(path):
    pts = []
    v_norms = []
    simps = []
    with open(path,"r") as f:
        for line in f:
            if line[:2] == 'v ':
                pts += [[float(i) for i in line.strip('\n').strip(' ').split(' ')[1:4]]]
            elif line[:2] == 'vn':
                norm = [float(i) for i in line.strip('\n').strip(' ').split(' ')[1:4]]
                v_norms += [ norm/np.linalg.norm(norm) ]
            elif line[:2] == 'f ':
                simps += [[ int(i.split('/')[0])-1 for i in line.strip('\n').strip(' ').split(' ')[1:4]]]
    return np.array(pts), np.array(v_norms), np.array(simps)


def read_obj_curve(path):
	pts = []
	simps = []
	with open(path,"r") as f:
		for line in f:
			if line[0] == 'v':
				pts += [[float(i) for i in line.strip('\n').strip(' ').split(' ')[1:4]]]
			elif line[0] == 'l':
				simps += [[ int(i)-1 for i in line.strip('\n').strip(' ').split(' ')[1:3]]]
	pts = np.array(pts)
	graph = {i:[] for i in range(len(simps))}
	for i,j in simps:
		graph[i].append(j)
		graph[j].append(i)
	out_indices = [0, graph[0][0]]
	while out_indices[-1] != 0:
		current = out_indices[-1]
		old = out_indices[-2]
		new = list(set(graph[current]).difference([old]))[0]
		out_indices.append(new)
	return pts[out_indices]

def make_curve2d(curve, noise=None):
    if noise:
        return np.array([curve.T[0], curve.T[2]]).T + noise*np.random.rand(len(curve),2)
    else:
        return np.array([curve.T[0], curve.T[2]]).T 

def distance(pair):
    """
    Calculate the Euclidean distance between two points.

    Args:
    point1 (array-like): Coordinates of the first point.
    point2 (array-like): Coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    point1, point2 = pair
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def plot_parametric_2d(x_func, y_func, t_range, num_points=1000, title="2D Parametric Plot"):
    """
    Plot a 2D parametric function.
    
    :param x_func: Function for x-coordinate
    :param y_func: Function for y-coordinate
    :param t_range: Tuple of (t_min, t_max) for parameter t
    :param num_points: Number of points to plot
    :param title: Title of the plot
    """
    t = np.linspace(t_range[0], t_range[1], num_points)
    x = x_func(t)
    y = y_func(t)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    return np.array([x,y]).T

def circle_through_three_points(p1, p2, p3):
    """
    Compute the center and radius of a circle passing through three points in the plane.
    
    :param p1: First point (x1, y1)
    :param p2: Second point (x2, y2)
    :param p3: Third point (x3, y3)
    :return: Tuple (center_x, center_y, radius)
    """
    # Convert input points to numpy arrays for easier computation
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Compute the perpendicular bisector of the line segment p1p2
    midpoint1 = (p1 + p2) / 2
    direction1 = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
    
    # Compute the perpendicular bisector of the line segment p2p3
    midpoint2 = (p2 + p3) / 2
    direction2 = np.array([-(p3[1] - p2[1]), p3[0] - p2[0]])
    
    # Solve the system of linear equations to find the intersection of the bisectors
    A = np.column_stack((direction1, -direction2))
    b = midpoint2 - midpoint1
    t = np.linalg.solve(A, b)
    
    # Compute the center of the circle
    center = midpoint1 + t[0] * direction1
    
    # Compute the radius
    radius = np.linalg.norm(center - p1)
    
    return center, radius

#probably not needed
def sphere_through_four_points(p1, p2, p3, p4):
    """
    Compute the center and radius of a sphere passing through four points in 3D space.
    
    :param p1: First point (x1, y1, z1)
    :param p2: Second point (x2, y2, z2) 
    :param p3: Third point (x3, y3, z3)
    :param p4: Fourth point (x4, y4, z4)
    :return: Tuple (center_x, center_y, center_z, radius)
    """
    # Convert input points to numpy arrays
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    
    # Set up the system of linear equations
    # For a sphere, (x - c)^2 + (y - c_y)^2 + (z - c_z)^2 = r^2 for each point
    # Subtracting equations eliminates r^2 term
    A = np.array([
        [2*(p2-p1)[0], 2*(p2-p1)[1], 2*(p2-p1)[2]],
        [2*(p3-p1)[0], 2*(p3-p1)[1], 2*(p3-p1)[2]],
        [2*(p4-p1)[0], 2*(p4-p1)[1], 2*(p4-p1)[2]]
    ])
    
    b = np.array([
        np.dot(p2,p2) - np.dot(p1,p1),
        np.dot(p3,p3) - np.dot(p1,p1),
        np.dot(p4,p4) - np.dot(p1,p1)
    ])
    
    # Solve for center
    center = np.linalg.solve(A, b)
    
    # Compute radius
    radius = np.sqrt(np.sum((p1 - center)**2))
    
    return center, radius


def evolute(pts, thresh, plot=True):
    centres = []
    for i in range(len(pts)):
        triple = pts[np.r_[(i-3):i]]
        circ = circle_through_three_points(*triple)
        if circ[1] < thresh:
             centres += [circ[0]]
    centres = np.array(centres)
    if plot==True:
        plt.figure(figsize=(10, 8))
        plt.plot(centres.T[0], centres.T[1],'r')
        plt.plot(pts.T[0], pts.T[1],'k')
        plt.title('evolute')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')
        plt.legend
        plt.legend(['Evolute', 'Original Curve'])
        plt.show()
    return centres

def create_mesh_from_arrays(x, y, z):
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Assuming x and y are 2D arrays defining a grid
    rows, cols = x.shape
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = i * cols + j   # Vertex indices start from 1 in OBJ files
            v2 = i * cols + j + 1
            v3 = (i + 1) * cols + j 
            v4 = (i + 1) * cols + j + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    return vertices, np.array(faces)

def evolute_3d(pts, simps, normals=None, radius=5):
    d1, d2, k1, k2 = igl.principal_curvature(pts, simps, radius=radius)
    if type(normals) == type(None):
        normals = igl.per_vertex_normals(pts, simps)
    focal1 = []
    focal2 = []
    for p, n, k1i, k2i in zip(pts, normals, k1, k2):
        focal1.append(p - n/k1i)
        focal2.append(p - n/k2i)
    focal1 = np.array(focal1)
    focal2 = np.array(focal2)
    return focal1, focal2

# incomplete
# Estimating Curvatures and Their Derivatives on Triangle Meshes
# Szymon Rusinkiewicz
def my_evolute_3d(pts, v_norms, tris, thresh):
    #calculate coordinate sys for each vertex (u_p, v_p)
    rand_vect = np.random.rand(1,3)
    v_coords = []
    for p, n_p in zip(pts, v_norms):
        u_p = np.cross(n_p, rand_vect)
        v_p = np.cross(n_p, u_p)
        v_coords += [[u_p, v_p]]
    for tri in tris:
        n_f = v_norms[tri].sum(axis=0)
        u_f = np.cross(n_f, rand_vect)
        v_f = np.cross(n_f, u_f)
        edge_vectors = []
        norm_deltas = []
        for (v1, v2) in list(it.combinations(tri, 2)):
            edge_vectors += [pts[v1] - pts[v2]]
            norm_deltas += [ v_norms[v1] - v_norms[v2]]
        x1 = np.dot(edge_vectors[0], u_f.T)[0]
        x2 = np.dot(edge_vectors[0], v_f.T)[0]
        y1 = np.dot(edge_vectors[1], u_f.T)[0]
        y2 = np.dot(edge_vectors[1], v_f.T)[0]
        z1 = np.dot(edge_vectors[2], u_f.T)[0]
        z2 = np.dot(edge_vectors[2], v_f.T)[0]

        a1 = np.dot(norm_deltas[0], u_f.T)[0]
        a2 = np.dot(norm_deltas[0], v_f.T)[0]
        b1 = np.dot(norm_deltas[0], u_f.T)[0]
        b2 = np.dot(norm_deltas[0], v_f.T)[0]
        c1 = np.dot(norm_deltas[0], u_f.T)[0]
        c2 = np.dot(norm_deltas[0], v_f.T)[0]

        A = np.array(
            [[x1,x2,0],
            [0,x1,x2],
            [y1,y2,0],
            [0,y1,y2],
            [z1,z2,0],
            [0,z1,z2]])
        b = np.array([a1,a2,b1,b2,c1,c2]).T
        x = np.linalg.lstsq(A,b)[0]
        fund_form2_f = np.array(
            [[x[0], x[1]],
            [x[1], x[2]]])
        #break
        for vert in tri:
            u_p, v_p = v_coords[vert]
            e_p = np.matmul(np.matmul(np.array([np.dot(u_p, u_f.T)[0][0], np.dot(u_p, v_f.T)[0][0]]), fund_form2_f), np.array([np.dot(u_p, u_f.T)[0][0], np.dot(u_p, v_f.T)[0][0]]).reshape([2,1]))[0]
            #double check u, v if something goes wrong
            f_p = np.matmul(np.matmul(np.array([np.dot(u_p, u_f.T)[0][0], np.dot(u_p, v_f.T)[0][0]]), fund_form2_f), np.array([np.dot(v_p, u_f.T)[0][0], np.dot(v_p, v_f.T)[0][0]]).reshape([2,1]))[0]
            g_p = np.matmul(np.matmul(np.array([np.dot(v_p, u_f.T)[0][0], np.dot(v_p, v_f.T)[0][0]]), fund_form2_f), np.array([np.dot(v_p, u_f.T)[0][0], np.dot(v_p, v_f.T)[0][0]]).reshape([2,1]))[0]
        break

def compute_2d_lambda_medial_axis(points, lambda_value, farthest=False):
    # Compute Voronoi diagram
    vor = Voronoi(points, furthest_site=farthest)

    # Get Voronoi vertices
    vertices = vor.vertices

    # Compute distances from each Voronoi vertex to its nearest input point
    distances = cdist(vertices, points).min(axis=1)

    # Get ridge vertices (edges) of the Voronoi diagram
    ridge_points = np.array(vor.ridge_points)

    # Filter edges that have distance between their defining points greater than lambda
    lambda_ma_points = []
    for edge in ridge_points:
        if -1 in edge:
            continue
        elif distance(points[edge]) >= lambda_value:
            lambda_ma_points.append(edge)
    lambda_ma_edges = np.array([vor.ridge_dict[tuple(i.tolist())] for i in lambda_ma_points])

    if len(lambda_ma_edges)>0:
        lambda_ma_vertices = lambda_ma_edges[:,0].tolist()
        lambda_ma_vertices.extend(lambda_ma_edges[:,1].tolist())
        lambda_ma_vertices = set(lambda_ma_vertices)
        lambda_ma_vertices = np.array(list(lambda_ma_vertices))
        lambda_ma_vertices = vertices[lambda_ma_vertices]
    else:
        lambda_ma_vertices = np.empty([0,3])
    return vertices, lambda_ma_vertices, lambda_ma_edges

# prints the evolute and medial axes for input of multiple curves
# first curve goes to pts additional (curve, lambda) pairs can be added to args
def plot_medial_evolute(pts, lambda_value, pt, outfile=None, col0='b', col1='r', *args):
    extra_curves = [ar[0] for ar in args]
    extra_lambdas = [ar[1] for ar in args]
    med_points = pts.tolist()
    if len(extra_curves) > 0:
        for curve in extra_curves:
            med_points +=  curve.tolist()
    med_points = np.array(med_points)

    vertices0, lambda_ma_vertices0, lambda_ma_edges0 = compute_2d_lambda_medial_axis(med_points, lambda_value, False)
    vertices1, lambda_ma_vertices1, lambda_ma_edges1 = compute_2d_lambda_medial_axis(med_points, lambda_value, True)
    lambda_ma_edges0_all = [lambda_ma_edges0]
    lambda_ma_edges1_all = [lambda_ma_edges1]
    vertices0_all = [vertices0]
    vertices1_all = [vertices1]
    vertices0, lambda_ma_vertices0, lambda_ma_edges0 = compute_2d_lambda_medial_axis(pts, lambda_value, False)
    vertices1, lambda_ma_vertices1, lambda_ma_edges1 = compute_2d_lambda_medial_axis(pts, lambda_value, True)
    lambda_ma_edges0_all += [lambda_ma_edges0]
    lambda_ma_edges1_all += [lambda_ma_edges1]
    vertices0_all += [vertices0]
    vertices1_all += [vertices1]
    for curve, lambda_i in zip(extra_curves, extra_lambdas):
        vertices0, lambda_ma_vertices0, lambda_ma_edges0 = compute_2d_lambda_medial_axis(curve, lambda_i, False)
        vertices1, lambda_ma_vertices1, lambda_ma_edges1 = compute_2d_lambda_medial_axis(curve, lambda_i, True)
        lambda_ma_edges0_all += [lambda_ma_edges0]
        lambda_ma_edges1_all += [lambda_ma_edges1]
        vertices0_all += [vertices0]
        vertices1_all += [vertices1]
    evos = []
    if len(extra_curves) > 0:
        evos = [evolute(curve, np.inf*lambda_value, plot=False) for curve in extra_curves]
    evos += [evolute(pts, np.inf*lambda_value, plot=False)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35, 15))
    
    max_dist = np.apply_along_axis(np.linalg.norm, 1, pts).max()
    for evo in evos:
        colour_range = [colorsys.hsv_to_rgb(x,1,1) for x in np.linspace(0,1,len(evo))]
        for i in range(len(evo)):
            if distance([evo[np.r_[i-1]], evo[np.r_[i]]]) > max_dist:
                continue
            else:
                ax1.plot(evo.T[0][np.r_[(i-2):i]], evo.T[1][np.r_[(i-2):i]], color=colour_range[i]) #, label='Evolute')
    '''
    jump_indices = [i for i in range(len(evo)) if distance((evo[i], evo[(i+1)%len(evo)])) > 5]
    k=0
    for j in jump_indices:
        ax1.plot(evo[k:(j)].T[0], evo[k:(j)].T[1], color='r')
        k=j
    '''
    colour_range = [colorsys.hsv_to_rgb(x,1,1) for x in np.linspace(0,1,len(pts))]
    for i in range(len(pts)):
        ax1.plot(pts.T[0][np.r_[(i-2):i]], pts.T[1][np.r_[(i-2):i]], color=colour_range[i])#, label='Original Curve')
    for curve in extra_curves:
        colour_range = [colorsys.hsv_to_rgb(x,1,1) for x in np.linspace(0,1,len(curve))]
        for i in range(len(curve)):
            ax1.plot(curve.T[0][np.r_[(i-2):i]], curve.T[1][np.r_[(i-2):i]], color=colour_range[i])#, label='Original Curve')
    ax1.scatter([pt[0]], [pt[1]], color='k', label='Filtration Point')
    # Plot lambda-MA vertices
    #plt.scatter(lambda_ma_vertices[:, 0], lambda_ma_vertices[:, 1], c='g', label='λ-MA Vertices')
    
    # Plot lambda-MA-0 edges
    i=0
    for j,(lambda_ma_edges0, vertices0) in enumerate(zip(lambda_ma_edges0_all, vertices0_all)):
        for edge in lambda_ma_edges0:
            if edge[0] != -1 and edge[1] != -1:  # Avoid infinite edges
                if i==0 and j==0:
                    ax1.plot(vertices0[edge][:, 0], vertices0[edge][:, 1], color=(.5,.5,.5), label='λ-MA (0)')
                    i=1
                else:
                    ax1.plot(vertices0[edge][:, 0], vertices0[edge][:, 1], color=(.5,.5,.5))
    # Plot lambda-MA-1 edges
    i=0
    for j,(lambda_ma_edges1, vertices1) in enumerate(zip(lambda_ma_edges1_all, vertices1_all)):
        for edge in lambda_ma_edges1:
            if edge[0] != -1 and edge[1] != -1:  # Avoid infinite edges
                if i==0 and j==0:
                    ax1.plot(vertices1[edge][:, 0], vertices1[edge][:, 1], color='k', label='λ-MA (1)')
                    i=1
                else:
                    ax1.plot(vertices1[edge][:, 0], vertices1[edge][:, 1], color='k')
    


    ax1.legend()
    ax1.set_title('2D λ-Medial Axes + Evolute')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend(loc = 'lower right')
    frac = .2
    ax1.set_xlim([pts.T[0].min()*(1 - frac*np.sign(pts.T[0].min())), pts.T[0].max()*(1 + frac*np.sign(pts.T[0].max()))])
    ax1.set_ylim([pts.T[1].min()*(1 - frac*np.sign(pts.T[1].min())), pts.T[1].max()*(1 + frac*np.sign(pts.T[1].max()))])



    # build extended Persistence diagram
    cplx = gudhi.SimplexTree()
    cplx.set_dimension(2)
    for i in range(len(pts)):
        cplx.insert([i], filtration=np.linalg.norm(pts[i] - np.array(pt)))
        cplx.insert([i, (i+1)%len(pts)], filtration=max([np.linalg.norm(pts[i] - np.array(pt)), np.linalg.norm(pts[(i+1)%len(pts)] - np.array(pt))]))
    counter = len(pts)
    for curve in extra_curves:
        for i in range(len(curve)):
            cplx.insert([i + counter], filtration=np.linalg.norm(curve[i] - np.array(pt)))
            cplx.insert([i + counter, (i+1)%len(curve) + counter], filtration=max([np.linalg.norm(curve[i] - np.array(pt)), np.linalg.norm(curve[(i+1)%len(curve)] - np.array(pt))]))
        counter += len(curve)
    cplx.extend_filtration()
    #cplx.compute_persistence(persistence_dim_max=2)
    dgms = cplx.extended_persistence()#persistence_dim_max=2)
    flat_dgms = [item for sublist in dgms for item in sublist]
    diag0 = np.array([ i[1] for i in flat_dgms if i[0]==0])
    diag1 = np.array([ i[1] for i in flat_dgms if i[0]==1])
    max_births, max_deaths = max([ i[1][0] for i in flat_dgms]), max([ i[1][1] for i in flat_dgms])
    max_diag = max(max_births, max_deaths)

    #gudhi.plot_persistence_diagram(cplx.persistence(), axes=ax2)
    ax2.plot([0, max_diag], [0, max_diag], color='k', label = 'Diagonal')
    ax2.scatter(diag0.T[0], diag0.T[1], color=col0, label='0th Diagram')
    ax2.scatter(diag1.T[0], diag1.T[1], color=col1, label='1st Diagram')
    ax2.legend(loc='lower right')
    ax2.set_title('Persistence Diagram')
    ax2.set_xlabel('Birth')
    ax2.set_ylabel('Death')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    #ax2.set_xlim([-0.1, 15])
    #ax2.set_ylim([-0.1, 15])
    if outfile==None:
         plt.show()
    else:
        fig.savefig(outfile, dpi=180)
    plt.close(fig)  # Close the figure to free up memory

    return dgms

# build extended Persistence diagram
def plot_extended_persistence3d(pts, simps, pt, outfile=None, col0='b', col1='r', col2='g'):
    if len(simps) > 0:
        d = max([len(i) for i in simps])
    else:
        d = 1
    cplx = gudhi.SimplexTree()
    cplx.set_dimension(d)
    for i, v in enumerate(pts):
        cplx.insert([i], filtration=np.linalg.norm(v - np.array(pt)))
    for simp in simps:
        cplx.insert(simp, filtration=max([cplx.filtration([i]) for i in simp]))
    #cplx.make_filtration_non_decreasing()
    cplx.extend_filtration()
    dgms = cplx.extended_persistence()#persistence_dim_max=2)
    
    flat_dgms = [item for sublist in dgms for item in sublist]
    diag0 = np.array([ i[1] for i in flat_dgms if i[0]==0])
    diag1 = np.array([ i[1] for i in flat_dgms if i[0]==1])
    diag2 = np.array([ i[1] for i in flat_dgms if i[0]==2])

    max_births, max_deaths = max([ i[1][0] for i in flat_dgms]), max([ i[1][1] for i in flat_dgms])
    max_diag = max(max_births, max_deaths)

    #gudhi.plot_persistence_diagram(cplx.persistence(), axes=ax2)
    plt.plot([0, max_diag], [0, max_diag], color='k', label = 'Diagonal')
    plt.scatter(diag0.T[0], diag0.T[1], color=col0, label='0th Diagram')
    if d>1:
        plt.scatter(diag1.T[0], diag1.T[1], color=col1, label='1st Diagram')
    if d>2:
        plt.scatter(diag2.T[0], diag2.T[1], color=col2, label='2nd Diagram')
    plt.legend(loc='lower right')
    #plt.set_title('Persistence Diagram')
    #plt.set_xlabel('Birth')
    #plt.set_ylabel('Death')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    #ax2.set_xlim([-0.1, 15])
    #ax2.set_ylim([-0.1, 15])
    if outfile==None:
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()
    return dgms

def dist(x, y):
    return np.sqrt( np.sum( np.square(y-x) ) )

def triangulate(face):
    tris = []
    for i in range(1,len(face)-1):
        tris += [[face[0], face[i], face[i+1]]]
    #tris += [[face[0],face[-1],face[1]]]
    return tris

def approx_medial_axis(input, MAX, z_max, alpha, LAMBDA, exp_tris, farthest):
    if type(input) == str:
        pts = read_obj(input)[0]
    else:
        pts = input
    colours = []
    pts = np.array(pts) + 1e-4*(np.random.rand(len(pts),3)-.5)
    
    
    voronoi = Voronoi(pts, furthest_site=farthest)
    print("done voronoi")
    delaunay = Delaunay(pts)
    print("done delaunay")
    
    dtris = set([])
    for i in delaunay.simplices:
        dtris.update([ j for j in it.combinations(i,3) if circumsphere_3d(pts[i])[1] < alpha])
    dtris = np.array([list(i) for i in dtris], dtype=np.int32)
    print("done alpha")
    
    bad_v = []
    if MAX:
        bad_v += [i for i,j in enumerate(voronoi.vertices) if dist(j,[0,0,0])>MAX]
    if z_max:
        bad_v += [i for i,j in enumerate(voronoi.vertices) if (np.abs(j[2]) > z_max)]
    
    bad_v = list(set(bad_v))
    
    faces = []
    for i in voronoi.ridge_dict:
        if -1 in voronoi.ridge_dict[i]:
            continue
        elif dist(pts[i[0]], pts[i[1]]) > LAMBDA:
            if len(np.intersect1d(voronoi.ridge_dict[i], bad_v, assume_unique = True)) == 0:
                if exp_tris:
                    faces += triangulate(voronoi.ridge_dict[i])
                else:
                    faces += [voronoi.ridge_dict[i]] 
    print("done lambda + limit")
    return [pts, dtris, voronoi.vertices, faces]

def area_tri(tri):
    a = dist(tri[0], tri[1])
    b = dist(tri[0], tri[2])
    c = dist(tri[1], tri[2])
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def gen_sample_from_obj(file, num_pts):
    pts, simps = read_obj(file)
    samples = []
    areas = np.array([area_tri(pts[i]) for i in simps])
    areas = areas/areas.sum()
    for simp_id in list(rd.choices(range(len(simps)), areas, k=num_pts)):
        simp = simps[simp_id]
        r1 = np.random.rand()
        r2 = np.random.rand()
        temp = (1 - np.sqrt(r1))*pts[simp[0]] + np.sqrt(r1)*(1 - r2)*pts[simp[1]] + r2*np.sqrt(r1)*pts[simp[2]]
        samples.append( temp )
    return np.array(samples)

def compute_normals_and_curvature(x_t, y_t, t_values):
    """
    Compute the vertex normals and curvature of a 2D curve.
    
    Parameters:
    - x_t: function representing the x-component of the curve as a function of t
    - y_t: function representing the y-component of the curve as a function of t
    - t_values: array of t-values at which to compute the normals and curvature
    
    Returns:
    - normals: Array of normal vectors at each t-value
    - curvatures: Array of curvatures at each t-value
    """
    # Initialize arrays for normals and curvatures

    
    # First derivative (tangent vector)
    dx_dts = np.gradient(np.apply_along_axis(x_t, 0, t_values))
    dy_dts = np.gradient(np.apply_along_axis(y_t, 0, t_values))

    # Second derivative (rate of change of the tangent vector)
    d2x_dt2 = np.gradient(dx_dts)
    d2y_dt2 = np.gradient(dy_dts)
    tangent = np.array([dx_dts, dy_dts]).T
    normal = np.array([-dy_dts, dx_dts]).T
    normal = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, normal)

    # Compute the curvature
    numerator = np.abs(dx_dts * d2y_dt2 - dy_dts * d2x_dt2)
    denominator = (dx_dts**2 + dy_dts**2)**(3/2)
    curvature = numerator / denominator
    
    return np.array(normal), np.array(curvature)

# there can be a problem if the link pinches in a vertex
def get_cyclic_link(vertex, network, faces, link_radius):
    
    star = {vertex}
    for i in range(link_radius):
        new_star = set([])
        for j in star:
            new_star.update(network[j])
        old_star = star
        star = new_star
    unordered_link = list(star.difference(old_star))

    if len(unordered_link) <= 3:
        return unordered_link
    else:
        ordered_link = [unordered_link[0]]
        current = unordered_link[0]
        next_verts = list(set(network[current]).intersection(unordered_link).difference(ordered_link))
        while len(next_verts) > 0:
            # necessary for coarse meshes, assume fine mesh
            '''
            if len(next_verts) > 1:
                for vert in next_verts:
                    if sorted([vertex, current, vert]) in faces:
                        ordered_link.append(vert)
                        current = vert
                        break
            else:
                ordered_link.append(next_verts[0])
                current = next_verts[0]
            '''
            ordered_link.append(next_verts[0])
            current = next_verts[0]
            next_verts = list(set(network[current]).intersection(unordered_link).difference(ordered_link))
        return ordered_link


# generate line bundle volume in 4d
'''
r = 5
pts, faces = read_obj("./ellipsoid3_test3.obj")
norms = igl.per_vertex_normals(pts, faces)
adj = igl.adjacency_list(faces)
d1, d2, k1, k2 = igl.principal_curvature(pts, faces, 5)
for i, pt in enumerate(pts):
    link_i = get_cyclic_link(i, adj, faces, 1)
    for j, pt_j in enumerate(link_i):
        D4_pt_0 = np.array(pt.tolist() + [0])
        D4_pt_1 = np.array(pts[link_i[j]] + [0])
        D4_pt_2 = np.array(pts[link_i[(j+1)%len(link_i)]] + [0])
        D4_pt_3 = np.array((pt + (r/k1[i])*norms[i]).tolist() + [(r/k1[i])])
        D4_pt_4 = np.array((pts[link_i[j]] + (r/k1[link_i[j]])*norms[link_i[j]]).tolist() + [(r/k1[link_i[j]])])
        D4_pt_5 = np.array((pts[link_i[(j+1)%len(link_i)]] + (r/k1[link_i[(j+1)%len(link_i)]])*norms[link_i[(j+1)%len(link_i)]]).tolist() + [(r/k1[link_i[(j+1)%len(link_i)]])])
'''










