from sym_set import *


##########################################################
# Start various figure/animation productions
##########################################################

############   Draw sym set for 2d curves

lambda_val = .4
n = 1000
frames = 150

#egg_x = lambda t: ((36 - np.sin(t)*np.sin(t))**(1 / 2) + np.cos(t))*np.cos(t)
egg_x = lambda t: ((23 - np.sin(t)*np.sin(t))**(1 / 2) + np.cos(t))*np.cos(t)
egg_y = lambda t: 4*np.sin(t)
pts_egg = plot_parametric_2d(egg_x, egg_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_egg += np.array([0,6])

'''
2 \cos \theta
1+3/2 \cos^2(\theta/2)) \sin(\theta)
'''
potato_x = lambda t: 2*np.cos(t)
potato_y = lambda t: (3/2)*np.cos(t/2)*np.cos(t/2)*np.sin(t)
#pts_potato = plot_parametric_2d(potato_x, potato_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_potato += np.array([0,6])


ellipse_x = lambda t: 6*np.cos(t)
ellipse_y = lambda t: 4*np.sin(t)
#pts_ellipse = plot_parametric_2d(ellipse_x, ellipse_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_ellipse -= np.array([0,6])

pts = np.array(pts_egg.tolist() + pts_ellipse.tolist())


bumpy_x = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.cos(np.pi*t)
bumpy_y = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.sin(np.pi*t)
#pts_bumpy = plot_parametric_2d(bumpy_x, bumpy_y, [0, 2], num_points=n, title="2D Parametric Plot")[:-1]


#plot_medial_evolute(pts_bumpy, lambda_val, [0.1, 0.1], None)
plot_medial_evolute(pts_egg, lambda_val, [0.1, 0.1], None)

#'''
# old animations
# make animation
for i,x in enumerate(np.linspace(-.5, .5, frames)):
    print(i)
    dgms = plot_medial_evolute(pts_egg, lambda_val, [3, x], './anim/frame_{}.png'.format(str(i).zfill(4)))
for i,x in enumerate(np.linspace(-1, 3, frames)):
    print(frames + i)
    dgms = plot_medial_evolute(pts_egg, lambda_val, [x, 8], './anim/frame_{}.png'.format(str(frames + i).zfill(4)))
os.sys("ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p extended_persistence.mp4")


circ_x = lambda t: np.cos(t)
circ_y = lambda t: np.sin(t)
pts_circ = plot_parametric_2d(circ_x, circ_y, [-np.pi, np.pi], num_points=frames, title="2D Parametric Plot")[:-1]
pts_circ += np.array([1.1, 1.5])
#'''

############   Animate sym set for evolving curves

lambda_val = .4
n = 1000
frames = 180

a = 1.5
b = 1
q = 20
T = np.linspace(-np.pi, np.pi, n+1)[:-1]
ptsX = [a*np.sin(t) for t in T]
ptsY = [b*(np.cos(t) - 1) for t in T]
for i, h in enumerate(np.linspace(0,0.15642458100558657,frames)):
    print(i)
    #pts = plot_parametric_2d(ptsX, ptsY, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
    H = [np.exp(-q*t*t)*h for t in T]
    pts = np.array([ptsX, ptsY]).T - np.array([np.zeros(n), H]).T
    plot_medial_evolute(pts, lambda_val, [0.1, 0.1], f"./evolving_anim/frame_{str(i).zfill(4)}.png")




############    make line bundle surface from curve

obj = read_obj_curve("./objs/test-curve1.obj")
n = len(obj)
t_values = np.linspace(0, 2*np.pi,n)
name_x = lambda t: curve.T[0][np.where(t_values == t)]
name_y = lambda t: curve.T[2][np.where(t_values == t)]


#circ_x = lambda t: 1.2*np.cos(t)
#circ_y = lambda t: 1*np.sin(t)
#obj = plot_parametric_2d(circ_x, circ_y, [0, 2*np.pi], num_points=n, title="2D Parametric Plot")[:-1]

#egg_x = lambda t: ((36 - np.sin(t)*np.sin(t))**(1 / 2) + np.cos(t))*np.cos(t)
#egg_y = lambda t: 4*np.sin(t)
#bumpy_x = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.cos(np.pi*t)
#bumpy_y = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.sin(np.pi*t)
#name_x, name_y = egg_x, egg_y
#n = 1000
l=100
obj = plot_parametric_2d(name_x, name_y, [0, 2*np.pi], num_points=n, title="2D Parametric Plot")[:-1]
normals, curvature = compute_normals_and_curvature(name_x, name_y, t_values)
e = evolute(obj, np.inf, False)
pts1, pts2, faces1, faces2 = [], [], [], []
for i, (pt, norm, curv) in enumerate(zip(obj, normals, curvature)):
    #print(i)
    #pt_0 = pt.tolist() + [0]
    #pt_1 = obj[(i+1)%(n-1)].tolist() + [0]
    pt_0 = (pt - (1/curv)*norm).tolist() + [-1/curv]
    pt_1 = (obj[(i+1)%(n-1)] - (1/curvature[(i+1)%(n-1)])*normals[(i+1)%(n-1)]).tolist() + [-1/curvature[(i+1)%(n-1)]]
    pt_2 = (obj[(i+1)%(n-1)] + (1/curvature[(i+1)%(n-1)])*normals[(i+1)%(n-1)]).tolist() + [1/curvature[(i+1)%(n-1)]]
    pt_3 = (pt + (1/curv)*norm).tolist() + [1/curv]
    pt_4 = ((l/curv)*norm + pt).tolist() + [l/curv]
    pt_5 = ((l/curvature[(i+1)%(n-1)])*normals[(i+1)%(n-1)] + obj[(i+1)%(n-1)]).tolist() + [l/curvature[(i+1)%(n-1)]]
    n1, n2 = i*4, i*4
    faces1.append([n1, n1+1, n1+2, n1+3])
    faces2.append([n2, n2+1, n2+2, n2+3])
    pts1 += [pt_0, pt_1, pt_2, pt_3]
    pts2 += [pt_2, pt_3, pt_4, pt_5]
write_obj("./objs/medial_0_faces.obj", pts1, faces1, "medial_0_faces")
write_obj("./objs/medial_1_faces.obj", pts2, faces2, "medial_1_faces")


############   2d example
lambda_val = .4
n = 1000
frames = 150
#circ_x = lambda t: (0.1+.05*t/(4*np.pi))*np.cos(t)
#circ_y = lambda t: (0.1+.05*t/(4*np.pi))*np.sin(t)
#spiral = plot_parametric_2d(circ_x, circ_y, [0, 4*np.pi], num_points=frames, title="2D Parametric Plot")[:-1]
#spiral += np.array([-0.174193, 0.162457])
circ_x = lambda t: 1*np.cos(t)
circ_y = lambda t: 1*np.sin(t)
circle = plot_parametric_2d(circ_x, circ_y, [0, 4*np.pi], num_points=frames, title="2D Parametric Plot")[:-1]
#circle += np.array([-0.174193, 0.162457])
# make animation
dgms_all = []
for i,x in enumerate(circle):
    print(i)
    dgms_all += [plot_medial_evolute(wormier, 0.1, x, './anim4/frame_{}.png'.format(str(i).zfill(4)), 'b', 'r')]
#os.system("ffmpeg -framerate 30 -pattern_type glob -i './anim4/*.png' -c:v libx264 -pix_fmt yuv420p extended_persistence_worm.mp4")



col0s = [colorsys.hsv_to_rgb(0.25,1,j) for j in np.linspace(0.5,0.9,frames)]
col1s = [colorsys.hsv_to_rgb(0.75,1,j) for j in np.linspace(0.5,0.9,frames)]
flat_dgms_all = [[item for sublist in dgm for item in sublist] for dgm in dgms_all]
max_births, max_deaths = max([max([ i[1][0] for i in flat_dgm]) for flat_dgm in flat_dgms_all]), max([max([ i[1][1] for i in flat_dgm]) for flat_dgm in flat_dgms_all])
max_diag = max(max_births, max_deaths)
plt.plot([0, max_diag], [0, max_diag], color='k', label = 'Diagonal')
diag0s = [] 
diag1s = []
for i,dgm in enumerate(dgms_all):
    print(i)
    flat_dgm = [item for sublist in dgm for item in sublist]
    diag0 = np.array([ i[1] for i in flat_dgm if i[0]==0])
    diag1 = np.array([ i[1] for i in flat_dgm if i[0]==1])
    diag0s += np.hstack((diag0, i*1e-1*np.ones([len(diag0),1]))).tolist()
    diag1s += np.hstack((diag1, i*1e-1*np.ones([len(diag1),1]))).tolist()
    plt.scatter(diag0.T[0], diag0.T[1], color=col0s[i])
    plt.scatter(diag1.T[0], diag1.T[1], color=col1s[i])
plt.axis('equal')
plt.grid(True)
plt.show()

diag0s = np.array(diag0s)
diag1s = np.array(diag1s)

visited = {}
alpha=.15
alpha_cmplx = []
dela = Delaunay(diag0s)
for tetra in dela.simplices:
    edges = it.combinations(tetra,2)
    for edge in edges:
        if tuple(sorted(list(edge))) in visited:
            continue
        else:
            verts = np.array(diag0s)[list(edge)]
            if distance(verts) < alpha:
                alpha_cmplx.append(tuple(sorted(list(edge))))
                visited[tuple(sorted(list(edge)))]=1
alpha_cmplx = np.array(alpha_cmplx)


write_obj('./objs/vineyard0.obj', diag0s, alpha_cmplx)
write_obj('./objs/vineyard1.obj', diag1s, [])


############   Create 3d egg
n = 100
u = np.linspace(-np.pi, np.pi, n)
v = np.linspace(-np.pi, np.pi, n)
uv = np.array(list(it.product(u,v)))
x = (1+0.2*uv[:,1])*np.cos(uv[:,0])*np.sin(uv[:,1])
y = (1+0.2*uv[:,1])*np.sin(uv[:,0])*np.sin(uv[:,1])
z = 1.65*np.cos(uv[:,1])
pts = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
write_obj('objs/egg2.obj', pts, [], 'egg')

############   Create 3d ellipsoid
n = 100
u = np.linspace(-np.pi, np.pi, n)
v = np.linspace(-np.pi, np.pi, n)
uv = np.array(list(it.product(u,v)))
a, b, c = 2/3, 1, 3/2 
x = a*np.sin(uv[:,0])*np.cos(uv[:,1])
y = b*np.sin(uv[:,0])*np.sin(uv[:,1])
z = c*np.cos(uv[:,0])
pts = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
write_obj('objs/new_ellipsoid.obj', pts, [], 'new_ellipsoid')

############   Create 3d 0th + 1st medial axes + evolute obj
name = "ellipsoid_dent"
file = "../sym_set/objs/{}.obj".format(name)
out_file = "../sym_set/objs/{}".format(name)

inputs = read_obj(file)
focal1, focal2 = evolute_3d(*inputs,radius=2)
write_obj("{}_focal1.obj".format(out_file), focal1, inputs[1], name="{}_focal_1".format(name))
write_obj("{}_focal2.obj".format(out_file), focal2, inputs[1], name="{}_focal_2".format(name))

alpha = 1
Lambda = .35
n = 10000
sample = gen_sample_from_obj(file, n)
#sample = read_obj(file)[0]   # for curves/knots/links
pts, tris, v_pts, v_faces = approx_medial_axis(sample, False, False, alpha, Lambda, True, False)
write_obj("{}_medial_0.obj".format(out_file), v_pts, v_faces, "{}_medial.0".format(name))
pts, tris, v_pts, v_faces = approx_medial_axis(sample, False, False, alpha, Lambda, True, True)
write_obj("{}_medial_2.obj".format(out_file), v_pts, v_faces, "{}_medial.2".format(name))


######   Create 3d evolute for saddle/extrema
x_bounds, xn = [-1,1] , 100
y_bounds, yn = [-1,1] , 100
x, y = np.meshgrid(np.linspace(*x_bounds, xn), np.linspace(*y_bounds, yn))
z = x**3 - 3*x*y**2
name = "monkey_saddle"
obj = create_mesh_from_arrays(x,y,z)
focal1, focal2 = evolute_3d(*obj,radius=2)
write_obj("./objs/{}.obj".format(name), *obj, name="{}".format(name))
write_obj("./objs/{}_focal1.obj".format(name), focal1, obj[1], name="{}_focal_1".format(name))
write_obj("./objs/{}_focal2.obj".format(name), focal2, obj[1], name="{}_focal_2".format(name))

######   Create 3d evolute for elliptic paraboloid
from sym_set import *
r, theta = x, y = np.meshgrid(np.linspace(0, .5, 50), np.linspace(0, 2*np.pi, 10))
x,y = r*np.cos(theta), r*np.sin(theta)
a, b = 3/2, 1
z = (x**2)/(a**2) + (y**2)/(b**2)
name = "elliptic_paraboloid"
obj = create_mesh_from_arrays(x,y,z)
focal1, focal2 = evolute_3d(*obj,radius=10)
write_obj("./objs/{}.obj".format(name), *obj, name="{}".format(name))
write_obj("./objs/{}_focal1.obj".format(name), focal1, obj[1], name="{}_focal_1".format(name))
write_obj("./objs/{}_focal2.obj".format(name), focal2, obj[1], name="{}_focal_2".format(name))

### e tries the dumbbell
name = "dumbbell"
obj = read_obj("./objs/dumbbell.obj")
focal1, focal2 = evolute_3d(*obj,radius=2)
write_obj("./objs/{}.obj".format(name), *obj, name="{}".format(name))
write_obj("./objs/{}_focal1.obj".format(name), focal1, obj[1], name="{}_focal_1".format(name))
write_obj("./objs/{}_focal2.obj".format(name), focal2, obj[1], name="{}_focal_2".format(name))


############   3d example
name = "twist_4"
#pts, simps = read_obj("objs/{}.obj".format(name))
pts = read_obj_curve(f"objs/{name}.obj")[:-1]
simps = np.array([[i,(i+1)%len(pts)] for i in range(len(pts))])
obsv_curve = read_obj_curve("objs/obsv_loop.obj")
lambda_val = .4
alpha = 5e-2
height = 0.016891891891891893
frames = len(obsv_curve)
#circ_x = lambda t: 0.125*np.cos(t)
#circ_y = lambda t: 0.125*np.sin(t)
#circle = plot_parametric_2d(circ_x, circ_y, [0, 6*np.pi], num_points=frames, title="2D Parametric Plot")[:-1]
#circle += np.array([-0.174193, 0.162457])
# make animation
dgms_all1 = []
dgms_all2 = []
for i,y in enumerate(obsv_curve):
    print(i)
    #x = np.array([y[0], y[1], 0])
    dgms_all1 += [plot_extended_persistence3d(pts1, simps1, y, None, 'b', 'r', 'g')]
    dgms_all2 += [plot_extended_persistence3d(pts2, simps2, y, None, 'b', 'r', 'g')]
#os.system("ffmpeg -framerate 30 -pattern_type glob -i './anim4/*.png' -c:v libx264 -pix_fmt yuv420p extended_persistence_worm.mp4")

col0s = [colorsys.hsv_to_rgb(0.,1,j) for j in np.linspace(0.5,0.9,frames)]
col1s = [colorsys.hsv_to_rgb(1/3,1,j) for j in np.linspace(0.5,0.9,frames)]
col2s = [colorsys.hsv_to_rgb(2/3,1,j) for j in np.linspace(0.5,0.9,frames)]
flat_dgms_all1 = [[item for sublist in dgm for item in sublist] for dgm in dgms_all1]
flat_dgms_all2 = [[item for sublist in dgm for item in sublist] for dgm in dgms_all2]
max_births1, max_deaths1 = max([max([ i[1][0] for i in flat_dgm]) for flat_dgm in flat_dgms_all1]), max([max([ i[1][1] for i in flat_dgm]) for flat_dgm in flat_dgms_all1])
max_diag1 = max(max_births1, max_deaths1)
plt.plot([0, max_diag1], [0, max_diag1], color='k', label = 'Diagonal')
diag0s = [] 
diag1s = []
diag2s = []
for j,dgms_all in enumerate([dgms_all1, dgms_all2]):
    print(j)
    for i,dgm in enumerate(dgms_all):
        print(i)
        flat_dgm = [item for sublist in dgm for item in sublist]
        diag0 = np.array([ i[1] for i in flat_dgm if i[0]==0])
        #diag1 = np.array([ i[1] for i in flat_dgm if i[0]==1])
        #diag2 = np.array([ i[1] for i in flat_dgm if i[0]==2])
        diag0s += np.hstack((diag0, i*height*np.ones([len(diag0),1]))).tolist()
        #diag1s += np.hstack((diag1, i*height*np.ones([len(diag1),1]))).tolist()
        #diag2s += np.hstack((diag2, i*height*np.ones([len(diag2),1]))).tolist()
        plt.scatter(diag0.T[0], diag0.T[1], color=col0s[i])
        #plt.scatter(diag1.T[0], diag1.T[1], color=col1s[i])
        #plt.scatter(diag2.T[0], diag2.T[1], color=col2s[i])
plt.axis('equal')
plt.grid(True)
plt.show()

diag0s = np.array(diag0s)
diag1s = np.array(diag1s)
diag2s = np.array(diag2s)

diagss = [diag0s]#, diag1s, diag2s]

deg = 0
suppress = 0
visited = {}
alpha_cmplx = []
diagXs = diagss[deg]
dela = Delaunay(diagXs)
write_obj("./objs/blah1.obj", diagXs, [], 'test')
for i, tetra in enumerate(dela.simplices):
    if int(i*100/len(dela.simplices))%5 == 0:
        if suppress == 0:
            print("{}%: ".format(int(i*100/len(dela.simplices))), tetra)
            suppress = 1
    else:
        suppress = 0
    edges = it.combinations(tetra,2)
    for edge in edges:
        if tuple(sorted(list(edge))) in visited:
            continue
        else:
            verts = np.array(diagXs)[list(edge)]
            if distance(verts) < alpha and np.abs(diagXs[edge[0]][2] - diagXs[edge[1]][2]) > 1e-6 and np.abs(diagXs[edge[0]][2] - diagXs[edge[1]][2]) < height+1e-6:
                alpha_cmplx.append(tuple(sorted(list(edge))))
                visited[tuple(sorted(list(edge)))]=1
alpha_cmplx = np.array(alpha_cmplx)


write_obj('./objs/{}_vines_{}.obj'.format(name, deg), diagXs, alpha_cmplx, "{}_vines_{}".format(name, deg))







################################################################
# make alpha complex
alpha = 0.05
dela = spa.Delaunay(sample)
combos = np.array([sorted(list(it.combinations(i,3))) for i in dela.simplices])
combos = combos.reshape((len(combos)*4,3)).tolist()
tris = [list(i) for i in set([ tuple(i) for i in [sorted(j) for j in combos]])]


dtris = np.array([ j for j in tris if circumsphere_3d(sample[j])[1] < alpha])
write_obj("../sym_set/objs/sample.obj", sample, dtris)







################################################################
# make 4d offset diagrams
################################################################
d = 4
cplx = gudhi.SimplexTree()
cplx.set_dimension(d)
pts = read_obj_curve("./objs/curve_fig3.obj")
pts = np.array([pts.T[0], pts.T[2], pts.T[1]]).T
tube_pts, tube_tets = create_tubular_nbhd_4d(pts, 0.05, 1)
for i, v in enumerate(tube_pts):
    cplx.insert([i], filtration=tube_pts[i][1])
for simp in tube_tets:
    cplx.insert(simp, filtration=max([cplx.filtration([i]) for i in simp]))
#cplx.make_filtration_non_decreasing()
cplx.extend_filtration()
dgms = cplx.extended_persistence()#persistence_dim_max=2)

flat_dgms = [item for sublist in dgms for item in sublist]
diag0 = np.array([ i[1] for i in flat_dgms if i[0]==0])
diag1 = np.array([ i[1] for i in flat_dgms if i[0]==1])
diag2 = np.array([ i[1] for i in flat_dgms if i[0]==2])
diag3 = np.array([ i[1] for i in flat_dgms if i[0]==3])
max_births, max_deaths = max([ i[1][0] for i in flat_dgms]), max([ i[1][1] for i in flat_dgms])
min_births, min_deaths = min([ i[1][0] for i in flat_dgms]), min([ i[1][1] for i in flat_dgms])
max_diag = max(max_births, max_deaths)
min_diag = min(min_births, min_deaths)

fig = plt.figure()
ax2 = fig.add_subplot()
col0, col1, col2, col3 = 'b', 'r', 'g', 'm'
ax2.plot([min_diag, max_diag], [min_diag, max_diag], color='k', label = 'Diagonal')
ax2.scatter(diag0.T[0], diag0.T[1], color=col0, label='0th Diagram')
ax2.scatter(diag1.T[0], diag1.T[1], color=col1, label='1st Diagram')
ax2.scatter(diag2.T[0], diag2.T[1], color=col2, label='2nd Diagram')
#ax2.scatter(diag3.T[0], diag3.T[1], color=col3, label='3rd Diagram')
ax2.legend(loc='lower right')
ax2.set_title('Persistence Diagram')
ax2.set_xlabel('Birth')
ax2.set_ylabel('Death')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()
ax2.set_xlim([min_diag, max_diag])
ax2.set_ylim([min_diag, max_diag])
plt.show()


