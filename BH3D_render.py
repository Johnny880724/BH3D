# solve geodesics for photons in Schwarzschild metric
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time

start_time = time.time()

# Constants
# G = 6.67430e-11  # Gravitational constant
# c = 3.0e8        # Speed of light
# M = 1.0e30       # Mass of the black hole (in kg)
c, G = 1, 1
M = 1
rs = 2 * G * M / c**2  # Schwarzschild radius

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm == 0, v, v / norm)

def cartesian_to_spherical(x3U_cart, u3U_cart):
    # Extract Cartesian position and velocity components
    x, y, z = x3U_cart[0], x3U_cart[1], x3U_cart[2]
    vx, vy, vz = u3U_cart[0], u3U_cart[1], u3U_cart[2]

    # Compute the radial distance r
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute the polar angle theta (colatitude)
    theta = np.arccos(z / r)

    # Compute the azimuthal angle phi (longitude)
    phi = np.arctan2(y, x)  # Handles quadrant correctly

    # Spherical position vector [r, theta, phi]
    x3U_sphe = [r, theta, phi]

    # Compute the radial velocity component (v_r)
    v_r = (x / r) * vx + (y / r) * vy + (z / r) * vz

    # Compute the polar velocity component (v_theta)
    v_theta = (x * z / (r**2 * np.sqrt(x**2 + y**2))) * vx \
            + (y * z / (r**2 * np.sqrt(x**2 + y**2))) * vy \
            - (np.sqrt(x**2 + y**2) / r**2) * vz

    # Compute the azimuthal velocity component (v_phi)
    v_phi = (-y / (x**2 + y**2)) * vx \
          + (x / (x**2 + y**2)) * vy

    # Spherical velocity vector [v_r, v_theta, v_phi]
    u3U_sphe = [v_r, v_theta, v_phi]

    return x3U_sphe, u3U_sphe



def g4DD_diag(x4U):
    t, rad, the, phi = x4U
    gDD00, gDD11, gDD22, gDD33 = -(1-rs/rad), 1/(1-rs/rad), rad**2, (rad**2 * np.sin(the)**2)
    return np.array([gDD00, gDD11, gDD22, gDD33])

def g4UU_diag(x4U):
    t, rad, the, phi = x4U
    gUU00, gUU11, gUU22, gUU33 = -1/(1-rs/rad), (1-rs/rad), 1/rad**2, 1/(rad**2 * np.sin(the)**2)
    return np.array([gUU00, gUU11, gUU22, gUU33])

# Geodesic equations for Schwarzschild metric
def geodesics(t, var):
    #conserved quantities
    #pt=-E, pphi=L
    # rad, the, phi, prad, pthe, pphi = y
    t, rad, the, phi, pt, prad, pthe, pphi = var
    gUU00, gUU11, gUU22, gUU33 = -1/(1-rs/rad), (1-rs/rad), 1/rad**2, 1/(rad**2 * np.sin(the)**2)
    # gDD00, gDD11, gDD22, gDD33 = -(1-rs/rad), 1/(1-rs/rad), rad**2, rad**2 * np.sin(the)**2
    d0gUU00, d0gUU11, d0gUU22, d0gUU33 = 0, 0, 0, 0
    d1gUU00, d1gUU11, d1gUU22, d1gUU33 = rs/rad**2 / (1-rs/rad)**2, rs/rad**2, -2/rad**3, -2/(rad**3 * np.sin(the)**2)
    d2gUU00, d2gUU11, d2gUU22, d2gUU33 = 0, 0, 0, -2*np.cos(the)/(rad**2*np.sin(the)**3)
    d3gUU00, d3gUU11, d3gUU22, d3gUU33 = 0, 0, 0, 0
    # pt = -np.sqrt(-(prad**2 * gUU11 + pthe**2 * gUU22 + pphi**2 * gUU33) / gUU00)
    # m2 = pt**2 * gUU00 + prad**2 * gUU11 + pthe**2 * gUU22 + pphi**2 * gUU33
    # print(m2)
    dtdt    = pt   * gUU00
    draddt  = prad * gUU11
    dthedt  = pthe * gUU22
    dphidt  = pphi * gUU33 
    dptdt   = -1/2 * (d0gUU00 * pt**2 + d0gUU11 * prad**2 + d0gUU22 * pthe**2 + d0gUU33 * pphi**2) # 0
    dpraddt = -1/2 * (d1gUU00 * pt**2 + d1gUU11 * prad**2 + d1gUU22 * pthe**2 + d1gUU33 * pphi**2)
    dpthedt = -1/2 * (d2gUU00 * pt**2 + d2gUU11 * prad**2 + d2gUU22 * pthe**2 + d2gUU33 * pphi**2)
    dpphidt = -1/2 * (d3gUU00 * pt**2 + d3gUU11 * prad**2 + d3gUU22 * pthe**2 + d3gUU33 * pphi**2) # 0
    # print([dtdt, draddt, dthedt, dphidt, dptdt, dpraddt, dpthedt, dpphidt])
    return [dtdt, draddt, dthedt, dphidt, dptdt, dpraddt, dpthedt, dpphidt]
    # return [draddt, dthedt, dphidt, dpraddt, dpthedt, dpphidt]

def interpolate_array(array, new_size):
    """
    Interpolates a 1D NumPy array to a new size.

    Parameters:
    array (np.ndarray): 1D NumPy array to be interpolated.
    new_size (int): The size of the output array.

    Returns:
    np.ndarray: Interpolated array of the new size.
    """
    # Create an interpolation function
    x = np.linspace(0, 1, len(array))
    f = interp1d(x, array, kind='linear')

    # Interpolate to the new size
    x_new = np.linspace(0, 1, new_size)
    interpolated_array = f(x_new)
    return interpolated_array

# Initial conditions
box_size = 20*rs
image_width, image_height = 400,300

pixel_to_unit = 0.01 / image_width
pixely_arr = np.linspace(-image_width / 2, image_width / 2, image_width)
pixelz_arr = np.linspace(-image_height / 2, image_height / 2, image_height)

traj_N = 1000

traj_x = np.zeros((image_width, image_height,traj_N))
traj_y = np.zeros((image_width, image_height,traj_N))
traj_z = np.zeros((image_width, image_height,traj_N))


for w,pixely in enumerate(pixely_arr):
    for h,pixelz in enumerate(pixelz_arr):
        # pixelx = image_width/2
        # pixely = image_height/2
        pixelx = -image_width /2
        ray_directions = np.array([pixelx, pixely, pixelz])
        
        coord_t0 = 0
        coord_x0 = 20*rs
        coord_y0 = 0*rs + pixely * pixel_to_unit
        coord_z0 = 0*rs + pixelz * pixel_to_unit
        x4U_cart = np.array([coord_t0, coord_x0, coord_y0, coord_z0])

        u3U_cart = normalize(ray_directions)
        x3U_sphe, u3U_sphe  = cartesian_to_spherical(x4U_cart[1:], u3U_cart)
        
        coord_rad0, coord_the0, coord_phi0    = x3U_sphe
        
        urad0, uthe0, uphi0                   = u3U_sphe

        x4U = np.array([coord_t0, coord_rad0, coord_the0, coord_phi0])
        u4U = np.array([0, urad0, uthe0, uphi0])
        ut0 = np.sqrt(-np.sum(u4U**2 * g4DD_diag(x4U))/g4DD_diag(x4U)[0])
        
        u4U = np.array([ut0, urad0, uthe0, uphi0])
        
        p4D = g4DD_diag(x4U) * u4U

        pt0, prad0, pthe0, pphi0 = p4D

        # Time span for the integration
        t_span = (0, 1000)
        # var0 = [rad0, the0, phi0, prad0, ptshe0, pphi0]
        # var0 = [t0, rad0, the0, phi0, pt0, prad0, pthe0, pphi0]
        var0 = np.hstack([x4U, p4D])

        # Solve the geodesic equations
        sol = solve_ivp(geodesics, t_span, var0, method='RK45', t_eval=np.linspace(t_span[0], t_span[1], 5000))

        # Extract the solution
        # rad = sol.y[0]
        # the = sol.y[1]
        # phi = sol.y[2]

        
        t, rad, the, phi, pt, prad, pthe, pphi = sol.y
        # Convert to Cartesian coordinates for plotting
        x = rad * np.sin(the) * np.cos(phi)
        y = rad * np.sin(the) * np.sin(phi)
        z = rad * np.cos(the)
        traj_x[w,h] = interpolate_array(x, traj_N)
        traj_y[w,h] = interpolate_array(y, traj_N)
        traj_z[w,h] = interpolate_array(z, traj_N)

print("Step 1: Trajectory generated")
        

def hit_ns(traj_x, traj_y, traj_z, ns_pos, ns_rad):
    # Calculate the distance from each point to the neutron star position
    distances = np.sqrt((traj_x - ns_pos[0])**2 + (traj_y - ns_pos[1])**2 + (traj_z - ns_pos[2])**2)
    
    # Determine if the distance is smaller than the neutron star radius
    hits = distances < ns_rad
    
    return hits

def hit_sky(traj_x, traj_y, traj_z, sky_pos, sky_rad):
    # Calculate the distance from each point to the neutron star position
    distances = np.sqrt((traj_x - sky_pos[0])**2 + (traj_y - sky_pos[1])**2 + (traj_z - sky_pos[2])**2)
    
    # Determine if the distance is smaller than the neutron star radius
    hits = distances > sky_rad
    return hits


sky_rad = 70*rs
NS_rad = 5*rs
BH_vis_rad = 1.1*rs

NS_pos = np.array([0,0,-5*rs])

hits_bh = hit_ns(traj_x, traj_y, traj_z, np.array([0,0,0]), BH_vis_rad)
hits_ns = hit_ns(traj_x, traj_y, traj_z, NS_pos, NS_rad)
hits_sky = hit_sky(traj_x, traj_y, traj_z, np.array([0,0,0]), sky_rad)

BH_mask  = np.any(hits_bh, axis=-1)
NS_mask  = np.any(hits_ns, axis=-1) * (1-BH_mask)
sky_mask = np.any(hits_sky, axis=-1) * (1-BH_mask) * (1-NS_mask)
NS_first_hit  = np.argmax(hits_ns , axis=-1)
sky_first_hit = np.argmax(hits_sky, axis=-1)

NS_hit_x = traj_x[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit]
NS_hit_y = traj_y[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit]
NS_hit_z = traj_z[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit]

sky_hit_x = traj_x[np.arange(image_width)[:, None],np.arange(image_height),sky_first_hit]
sky_hit_y = traj_y[np.arange(image_width)[:, None],np.arange(image_height),sky_first_hit]
sky_hit_z = traj_z[np.arange(image_width)[:, None],np.arange(image_height),sky_first_hit]

NS_hit_vx = NS_hit_x - traj_x[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit-1]
NS_hit_vy = NS_hit_y - traj_y[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit-1]
NS_hit_vz = NS_hit_z - traj_z[np.arange(image_width)[:, None],np.arange(image_height),NS_first_hit-1]

NS_hit_nx = NS_hit_x - NS_pos[0]
NS_hit_ny = NS_hit_y - NS_pos[1]
NS_hit_nz = NS_hit_z - NS_pos[2]

NS_intensity = - (NS_hit_vx * NS_hit_nx + NS_hit_vy * NS_hit_ny + NS_hit_vz * NS_hit_nz)\
        /(np.sqrt(NS_hit_vx**2 + NS_hit_vy**2 + NS_hit_vz**2)+1e-15)\
        /(np.sqrt(NS_hit_nx**2 + NS_hit_ny**2 + NS_hit_nz**2+1e-15))

print("Step 2: NS and sky hit cleared")

import random
# Generate random stars on the background shell
def generate_stars(num_stars, sky_rad):
    star_pos = np.zeros((num_stars,3))
    star_size = np.zeros(num_stars)
    for i in range(num_stars):
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        x = sky_rad*np.sin(phi) * np.cos(theta)
        y = sky_rad*np.sin(phi) * np.sin(theta)
        z = sky_rad*np.cos(phi)
        star_size[i] = 10**random.uniform(-10, -4)  # Random size for the star
        # star_size[i] = 10**random.uniform(-6, -4)  # Random size for the star
        star_pos[i]  = np.array([x,y,z])
    return star_pos, star_size

num_stars = 5000
star_pos, star_size = generate_stars(num_stars,sky_rad)
sky_hit_pos = np.stack((sky_hit_x,sky_hit_y,sky_hit_z),axis=-1)
hit_star_all = np.zeros((image_width,image_height,num_stars))
for i in range(num_stars):
    cos_angle = np.dot(sky_hit_pos,star_pos[i,:]) / (sky_rad*np.sqrt(np.sum(sky_hit_pos**2,axis=-1)))
    hit_star_all[:,:,i] = cos_angle > (1-star_size[i])

star_mask = np.any(hit_star_all,axis=-1)

print("Step 3: Stars hit cleared")

plt.figure(figsize=(40,30))

final_color = np.zeros((image_width,image_height,3))
NS_color_dark     = np.array([60, 90, 255])/255
NS_color_bright   = np.array([255, 255, 255])/255
BH_color   = np.array([0,0,0])
star_color = np.array([1,1,1])
final_color = NS_color_bright * NS_intensity[...,None] * NS_mask[...,None] \
            + NS_color_dark * NS_mask[...,None] * (1-NS_intensity[...,None]) \
            + BH_color * BH_mask[...,None] \
            + star_color * star_mask[...,None] * sky_mask[...,None] 

image = np.transpose(final_color, (1, 0, 2))
plt.imshow(np.clip(image, 0, 1))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

plt.savefig("bh3d.png",bbox_inches='tight')
print("Step 4: Image saved")
end_time = time.time()
print("time elapsed %f sec" % (end_time - start_time))