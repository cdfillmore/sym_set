from sym_set import *


##########################################################
# Start various figure/animation productions
##########################################################

############   Draw sym set for 2d curves

lambda_val = .4
n = 1000
frames = 150

egg_x = lambda t: ((36 - np.sin(t)*np.sin(t))**(1 / 2) + np.cos(t))*np.cos(t)
egg_y = lambda t: 4*np.sin(t)
pts_egg = plot_parametric_2d(egg_x, egg_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_egg += np.array([0,6])

'''
2 \cos \theta
1+3/2 \cos^2(\theta/2)) \sin(\theta)
'''
potato_x = lambda t: 2*np.cos(t)
potato_y = lambda t: (3/2)*np.cos(t/2)*np.cos(t/2)*np.sin(t)
pts_potato = plot_parametric_2d(potato_x, potato_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_potato += np.array([0,6])


ellipse_x = lambda t: 6*np.cos(t)
ellipse_y = lambda t: 4*np.sin(t)
pts_ellipse = plot_parametric_2d(ellipse_x, ellipse_y, [-np.pi, np.pi], num_points=n, title="2D Parametric Plot")[:-1]
pts_ellipse -= np.array([0,6])

pts = np.array(pts_egg.tolist() + pts_ellipse.tolist())


bumpy_x = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.cos(np.pi*t)
bumpy_y = lambda t: (1.5 + .5*np.sin(6*np.pi*t))*np.sin(np.pi*t)
pts_bumpy = plot_parametric_2d(bumpy_x, bumpy_y, [0, 2], num_points=n, title="2D Parametric Plot")[:-1]


plot_medial_evolute(pts_bumpy, lambda_val, [0.1, 0.1], None)
plot_medial_evolute(pts_egg, lambda_val, [0.1, 0.1], None)

'''
# old animations
# make animation
for i,x in enumerate(np.linspace(8, 4, frames)):
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
'''



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

############   Creat 3d 0th + 1st medial axes + evolute obj
name = "thickened_moebius_smoothed"
file = "../sym_set/objs/{}.obj".format(name)
out_file = "../sym_set/objs/{}".format(name)

input = read_obj(file)
focal1, focal2 = evolute_3d(*input,radius=2)
write_obj("{}_focal1.obj".format(file), focal1, input[1], name="{}_focal_1".format(name))
write_obj("{}_focal2.obj".format(file), focal2, input[1], name="{}_focal_2".format(name))

alpha = 1
Lambda = .8
n = 1000
sample = gen_sample_from_obj(file, n)
#sample = read_obj(file)[0]   # for curves/knots/links
pts, tris, v_pts, v_faces = approx_medial_axis(sample, False, False, alpha, Lambda, True, False)
write_obj("{}_medial_0.obj".format(out_file), v_pts, v_faces, "{}_medial_0".format(name))
pts, tris, v_pts, v_faces = approx_medial_axis(sample, False, False, alpha, Lambda, True, True)
write_obj("{}_medial_1.obj".format(out_file), v_pts, v_faces, "{}_medial_1".format(name))


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


############   3d example
pts, simps = read_obj("./objs/moebius_boundary_2.obj")
lambda_val = .4
alpha = 1e-2
height = 1e-3
frames = 500
circ_x = lambda t: 0.125*np.cos(t)
circ_y = lambda t: 0.125*np.sin(t)
circle = plot_parametric_2d(circ_x, circ_y, [0, 6*np.pi], num_points=frames, title="2D Parametric Plot")[:-1]
#circle += np.array([-0.174193, 0.162457])
# make animation
dgms_all = []
for i,y in enumerate(circle):
    print(i)
    x = np.array([y[0], y[1], 0])
    dgms_all += [plot_extended_persistence3d(pts, simps, x, None, 'b', 'r', 'g')]
#os.system("ffmpeg -framerate 30 -pattern_type glob -i './anim4/*.png' -c:v libx264 -pix_fmt yuv420p extended_persistence_worm.mp4")

col0s = [colorsys.hsv_to_rgb(0.,1,j) for j in np.linspace(0.5,0.9,frames)]
col1s = [colorsys.hsv_to_rgb(1/3,1,j) for j in np.linspace(0.5,0.9,frames)]
col2s = [colorsys.hsv_to_rgb(2/3,1,j) for j in np.linspace(0.5,0.9,frames)]
flat_dgms_all = [[item for sublist in dgm for item in sublist] for dgm in dgms_all]
max_births, max_deaths = max([max([ i[1][0] for i in flat_dgm]) for flat_dgm in flat_dgms_all]), max([max([ i[1][1] for i in flat_dgm]) for flat_dgm in flat_dgms_all])
max_diag = max(max_births, max_deaths)
plt.plot([0, max_diag], [0, max_diag], color='k', label = 'Diagonal')
diag0s = [] 
diag1s = []
diag2s = []
for i,dgm in enumerate(dgms_all):
    print(i)
    flat_dgm = [item for sublist in dgm for item in sublist]
    diag0 = np.array([ i[1] for i in flat_dgm if i[0]==0])
    diag1 = np.array([ i[1] for i in flat_dgm if i[0]==1])
    diag2 = np.array([ i[1] for i in flat_dgm if i[0]==2])
    diag0s += np.hstack((diag0, i*height*np.ones([len(diag0),1]))).tolist()
    diag1s += np.hstack((diag1, i*height*np.ones([len(diag1),1]))).tolist()
    diag2s += np.hstack((diag2, i*height*np.ones([len(diag2),1]))).tolist()
    plt.scatter(diag0.T[0], diag0.T[1], color=col0s[i])
    plt.scatter(diag1.T[0], diag1.T[1], color=col1s[i])
    plt.scatter(diag2.T[0], diag2.T[1], color=col2s[i])
plt.axis('equal')
plt.grid(True)
plt.show()

diag0s = np.array(diag0s)
diag1s = np.array(diag1s)
diag2s = np.array(diag2s)

suppress = 0
visited = {}
alpha_cmplx = []
diagXs = diag2s
dela = Delaunay(diagXs)
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


write_obj('./objs/vineyard2_3d.obj', diagXs, alpha_cmplx)







################################################################
# make alpha complex
alpha = 0.05
dela = spa.Delaunay(sample)
combos = np.array([sorted(list(it.combinations(i,3))) for i in dela.simplices])
combos = combos.reshape((len(combos)*4,3)).tolist()
tris = [list(i) for i in set([ tuple(i) for i in [sorted(j) for j in combos]])]


dtris = np.array([ j for j in tris if circumsphere_3d(sample[j])[1] < alpha])
write_obj("../sym_set/objs/sample.obj", sample, dtris)














