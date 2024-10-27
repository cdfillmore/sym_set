import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import gudhi

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

def evolute(pts, plot=True):
    centres = []
    for i in range(len(pts)):
        triple = pts[np.r_[(i-3):i]]
        centres += [circle_through_three_points(*triple)[0]]
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

    lambda_ma_vertices = lambda_ma_edges[:,0].tolist()
    lambda_ma_vertices.extend(lambda_ma_edges[:,1].tolist())
    lambda_ma_vertices = set(lambda_ma_vertices)
    lambda_ma_vertices = np.array(list(lambda_ma_vertices))
    lambda_ma_vertices = vertices[lambda_ma_vertices]
    
    return vertices, lambda_ma_vertices, lambda_ma_edges

def plot_medial_evolute(pts, lambda_value, pt, outfile):
    vertices0, lambda_ma_vertices0, lambda_ma_edges0 = compute_2d_lambda_medial_axis(pts, lambda_value, False)
    vertices1, lambda_ma_vertices1, lambda_ma_edges1 = compute_2d_lambda_medial_axis(pts, lambda_value, True)
    evo = evolute(pts, plot=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35, 15))
    
    ax1.plot(evo.T[0], evo.T[1],'r', label='Evolute')
    ax1.plot(pts.T[0], pts.T[1],'k', label='Original Curve')
    ax1.scatter([pt[0]], [pt[1]], color='k', label='Filtration Point')
    # Plot lambda-MA vertices
    #plt.scatter(lambda_ma_vertices[:, 0], lambda_ma_vertices[:, 1], c='g', label='λ-MA Vertices')
    
    # Plot lambda-MA-0 edges
    for i,edge in enumerate(lambda_ma_edges0):
        if edge[0] != -1 and edge[1] != -1:  # Avoid infinite edges
            if i==0:
                ax1.plot(vertices0[edge][:, 0], vertices0[edge][:, 1], 'g-', label='λ-MA (0)')
            else:
                ax1.plot(vertices0[edge][:, 0], vertices0[edge][:, 1], 'g-')
    # Plot lambda-MA-1 edges
    for i,edge in enumerate(lambda_ma_edges1):
        if edge[0] != -1 and edge[1] != -1:  # Avoid infinite edges
            if i==0:
                ax1.plot(vertices1[edge][:, 0], vertices1[edge][:, 1], 'm-', label='λ-MA (1)')
            else:
                ax1.plot(vertices1[edge][:, 0], vertices1[edge][:, 1], 'm-')
    
    ax1.legend()
    ax1.set_title('2D λ-Medial Axes + Evolute')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    cplx = gudhi.SimplexTree()
    cplx.set_dimension(2)
    for i in range(len(pts)):
        cplx.insert([i], filtration=np.linalg.norm(pts[i] - np.array(pt)))
        cplx.insert([i, (i+1)%len(pts)], filtration=max([np.linalg.norm(pts[i] - np.array(pt)), np.linalg.norm(pts[(i+1)%len(pts)] - np.array(pt))]))
    cplx.compute_persistence(persistence_dim_max=2)
    gudhi.plot_persistence_diagram(cplx.persistence(), axes=ax2)
    fig.savefig(outfile)
    plt.close(fig)  # Close the figure to free up memory
    pass



lambda_val = .1
n = 1000

egg_x = lambda t: ((36 - np.sin(t)*np.sin(t))**(1 / 2) + np.cos(t))*np.cos(t)
egg_y = lambda t: 4*np.sin(t)
pts_egg = plot_parametric_2d(egg_x, egg_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_egg += np.array([0,6])

ellipse_x = lambda t: 6*np.cos(t)
ellipse_y = lambda t: 4*np.sin(t)
pts_ellipse = plot_parametric_2d(ellipse_x, ellipse_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_ellipse -= np.array([0,6])

pts = np.array(pts_egg.tolist() + pts_ellipse.tolist())


# make animation
for i,x in enumerate(np.linspace(8, 4, 300)):
    print(i)
    plot_medial_evolute(pts_egg, lambda_val, [3, x], './anim/frame_{}.png'.format(str(i).zfill(4)))










