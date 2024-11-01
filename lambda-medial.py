import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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


def compute_2d_lambda_medial_axis(points, lambda_value):
    # Compute Voronoi diagram
    vor = Voronoi(points)

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

    lambda_ma_vertices = lambda_ma_edges[:,0].tolist()
    lambda_ma_vertices.extend(lambda_ma_edges[:,1].tolist())
    lambda_ma_vertices = set(lambda_ma_vertices)
    lambda_ma_vertices = np.array(list(lambda_ma_vertices))
    lambda_ma_vertices = vertices[lambda_ma_vertices]
    
    return vertices, lambda_ma_vertices, lambda_ma_edges

def plot_lambda_medial_axis(points, vertices, lambda_ma_vertices, lambda_ma_edges):
    plt.figure(figsize=(10, 10))
    
    # Plot input points
    plt.scatter(points[:, 0], points[:, 1], c='b', label='Input Points')
    
    # Plot lambda-MA vertices
    plt.scatter(lambda_ma_vertices[:, 0], lambda_ma_vertices[:, 1], c='r', label='λ-MA Vertices')
    
    # Plot lambda-MA edges
    for edge in lambda_ma_edges:
        if edge[0] != -1 and edge[1] != -1:  # Avoid infinite edges
            plt.plot(vertices[edge][:, 0], vertices[edge][:, 1], 'g-')
    
    plt.legend()
    plt.title('2D λ-Medial Axis')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# Generate points in the shape of an ellipse with noise
def generate_ellipse_points(num_points=100, a=1, b=0.5, noise_level=0.05):
    t = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Add noise
    noise_x = np.random.normal(0, noise_level, num_points)
    noise_y = np.random.normal(0, noise_level, num_points)
    
    x += noise_x
    y += noise_y
    
    return np.column_stack((x, y))

noise_level = 0.0
lambda_value = 0.05
num_points = 200

# Generate ellipse points
#shape_points = generate_ellipse_points(num_points=num_points, a=.5, b=.25, noise_level=noise_level)
shape_points = read_obj("S.obj")[0][:,[0,2]] + np.random.normal(0,noise_level,size=shape_points.shape)

# Use the generated shape points instead of the example points
points = shape_points.tolist()
points += (shape_points+np.array([2*np.pi,0])).tolist()
points += (shape_points+np.array([-2*np.pi,0])).tolist()
points += [[0,3],[0,-3.]]
points += [[2*np.pi,3],[2*np.pi,-3.]]
points += [[-2*np.pi,3],[-2*np.pi,-3.]]
points = np.array(points)
vertices, lambda_ma_vertices, lambda_ma_edges = compute_2d_lambda_medial_axis(points, lambda_value)
plot_lambda_medial_axis(points, vertices,lambda_ma_vertices, lambda_ma_edges)

col0 = vertices[:,0]
bad_inds = np.where(col0>np.pi)[0].tolist()
bad_inds += np.where(col0<-np.pi)[0].tolist()

good_edges = np.array([i for i in lambda_ma_edges if len(np.intersect1d(i, bad_inds))==0])
R3_data = np.array([np.exp(1j*vertices[:,0]).real, np.exp(1j*vertices[:,0]).imag, vertices[:,1]]).T
R3_shape_points = np.array([np.exp(1j*shape_points[:,0]).real, np.exp(1j*shape_points[:,0]).imag, shape_points[:,1]]).T

write_obj("lambda-ma-S.obj", R3_data, good_edges)
write_obj("S-shape.obj", R3_shape_points, read_obj("S.obj")[1])
