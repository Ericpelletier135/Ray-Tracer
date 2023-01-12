# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
from gpytoolbox import read_mesh, per_vertex_normals, per_face_normals # just used to load a mesh, now
##########################################


def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; they must have the same 
        size for gpytoolbox
        """
        if Os.shape[0] != Ds.shape[0]:
            if Ds.shape[0] == 1:
                self.Os = np.copy(Os)
                self.Ds = np.copy(Os)
                self.Ds[:, :] = Ds[:, :]
            if Os.shape[0] == 1:
                self.Ds = np.copy(Ds)
                self.Os = np.copy(Ds)
                self.Os[:, :] = Os[:, :]
        else:
            self.Os = np.copy(Os)
            self.Ds = np.copy(Ds)

    def __call__(self, t):
        """
        Computes an array of 3D locations given the distances
        to the points.
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return


# ===================== our replacement for gpytoolbox routines =====================
def get_bary_coords(intersection, tri):
    denom = area(tri[:, 0], tri[:, 1], tri[:, 2])
    alpha_numerator = area(intersection, tri[:, 1], tri[:, 2])
    beta_numerator = area(intersection, tri[:, 0], tri[:, 2])
    alpha = alpha_numerator / denom
    beta = beta_numerator / denom
    gamma = 1 - alpha - beta
    barys = np.vstack((alpha, beta, gamma)).transpose()
    barys = np.where(np.isnan(barys), 0, barys)
    return barys

def area(t0, t1, t2):
    n = np.cross(t1 - t0, t2 - t0, axis = 1)
    return np.linalg.norm(n, axis = 1) / 2

def ray_mesh_intersect(origin, dir, tri):
    intersection = np.ones_like(dir) * -1
    intersection[:, 2] = np.Inf
    dir = dir[:, None]
    
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0] # (num_triangles, 3)
    s = origin[:, None] - tri[:, 0][None]
    s1 = np.cross(dir, e2)
    s2 = np.cross(s, e1)
    s1_dot_e1 = np.sum(s1 * e1, axis=2)
    results = np.ones((dir.shape[0], tri.shape[0])) * np.Inf

    if (s1_dot_e1 != 0).sum() > 0:
        coefficient = np.reciprocal(s1_dot_e1)
        alpha = coefficient * np.sum(s1 * s, axis=2)
        beta = coefficient * np.sum(s2 * dir, axis=2)
        cond_bool = np.logical_and(
                        np.logical_and(
                            np.logical_and(0 <= alpha,  alpha < 1),
                            np.logical_and(0 <= beta,  beta < 1)
                        ),
                    np.logical_and(0 <= alpha + beta,  alpha + beta < 1)
            ) # (num_rays, num_tri)
        e1_expanded = np.tile(e1[None], (dir.shape[0], 1, 1)) # (num_rays, num_tri, 3)
        dot_temp = np.sum(s1[cond_bool] * e1_expanded[cond_bool], axis = 1) # (num_rays,)
        results_cond1 = results[cond_bool]
        cond_bool2 = dot_temp != 0 

        if cond_bool2.sum() > 0:
              coefficient2 = np.reciprocal(dot_temp)
              e2_expanded = np.tile(e2[None], (dir.shape[0], 1, 1)) # (num_rays, num_tri, 3)
              t = coefficient2 * np.sum( s2[cond_bool][cond_bool2] *
                                         e2_expanded[cond_bool][cond_bool2],
                                         axis = 1)
              results_cond1[cond_bool2] = t
        results[cond_bool] = results_cond1
    results[results <= 0] = np.Inf
    hit_id = np.argmin(results, axis=1)
    min_val = np.min(results, axis=1)
    hit_id[min_val == np.Inf] = -1
    return min_val, hit_id
# ===================== our replacement for gpytoolbox routines =====================

class Mesh(Geometry):
    def __init__(self, filename, brdf_params = np.array([0,0,0,1]), Le = np.array([0,0,0])):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        self.Le = Le
        ### BEGIN CODE
        self.face_normals = per_face_normals(self.v, self.f, unit_norm = True)
        self.vertex_normals = per_vertex_normals(self.v, self.f)
        ### END CODE
        super().__init__()

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])
        
        hit_distances, triangle_hit_ids = ray_mesh_intersect(rays.Os, rays.Ds, self.v[self.f])
        intersections = rays.Os + hit_distances[:, None] * rays.Ds
        tris = self.v[self.f[triangle_hit_ids]]
        barys = get_bary_coords(intersections, tris)

        ## BEGIN CODE
        faces = self.f[triangle_hit_ids]
        
        # get the vertices corresponding to each face
        temp_normals = self.vertex_normals[faces]

        barys = barys[:, :, np.newaxis]

        # multiply the vertices by the barys coordinates and then add them up
        temp_normals = barys * temp_normals
        temp_normals = np.sum(temp_normals, axis=1) 
        ### END CODE

        temp_normals = np.where( (triangle_hit_ids == -1)[:, np.newaxis],
                                 hit_normals,
                                 temp_normals )
        hit_normals = temp_normals

        return hit_distances, hit_normals


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params = np.array([0,0,0,1]), Le = np.array([0,0,0])):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays: output the
        intersection distances (set to np.inf if none), and unit hit
        normals (set to [np.inf, np.inf, np.inf] if none.)
        """

        ### BEGIN CODE
        A = np.sum(np.square(rays.Ds), axis=1)

        # calculate B in quadratic formula
        B = 2 * np.sum(rays.Ds * (rays.Os - self.c), axis=1)

        # calculate C in quadratic formula
        C = np.sum(np.square(rays.Os - self.c), axis=1) - np.square(self.r)

        # calculate discriminants
        deltas = np.square(B) - 4 * A * C

        # set np.inf where the discriminant is less than 0
        distances = np.where(deltas < 0, np.inf, deltas)

        # calculate the first and second intersection of all the rays (even if they dont exist) (set negative t's to np.inf)
        first_intersection = (-B + np.sqrt(np.square(B) - 4 * A * C)) / (2*A)
        first_intersection = np.where(first_intersection < Sphere.EPSILON_SPHERE, np.inf, first_intersection)

        second_intersection = (-B - np.sqrt(np.square(B) - 4 * A * C)) / (2*A)
        second_intersection = np.where(second_intersection < Sphere.EPSILON_SPHERE, np.inf, second_intersection)

        # calculate which intersection is smaller than the other
        min_intersection = np.where(first_intersection < second_intersection, first_intersection, second_intersection)

        # set the smallest intersection where the discriminant is bigger than 0
        distances = np.where(deltas > 0, min_intersection, distances)
        
        # set the intersection equal to t where the discriminant is equal to 0
        one_intersections = -B / (2*A)
        one_intersections = np.where(one_intersections < Sphere.EPSILON_SPHERE, np.inf, -B / (2*A))
        distances = np.where(deltas == 0, one_intersections, distances)

        # calculate hit points on sphere
        hit_points = rays.Ds * distances[:, np.newaxis] + rays.Os

        # calculate the normal at the hit point
        normals = hit_points - self.c
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)    

        ### END CODE

        return distances, normals


# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, LIGHT_SAMPLING, BRDF_SAMPLING, MIS_SAMPLING = range(4)


class Scene(object):
    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_geometries(self, geometries):
        """ 
        Adds a list of geometries to the scene.
        
        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([ geometries[i] ])

        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter = False):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """

        ### BEGIN CODE: feel free to remove any/all of our placeholder code, below
        yx = np.indices((self.h, self.w))
        
        x_indices = yx[1, :, :]
        y_indices = yx[0, :, :]

        if(jitter):
            x_NDC = (x_indices + np.random.uniform(0., 1., x_indices.shape)) / self.w
            y_NDC = (y_indices + np.random.uniform(0., 1., y_indices.shape)) / self.h
        else:
            x_NDC = (x_indices + 0.5) / (self.w)
            y_NDC = (y_indices + 0.5) / self.h  

        # convert x and y values for 2D screen with range (-1,1)
        x_Screen = 2 * x_NDC - 1
        y_Screen = 1 - 2 * y_NDC

        sceneRatio = self.w / self.h

        fov = np.tan((self.fov / 2) * (np.pi/180))

        # convert x and y values for the camera space 
        x_camera = x_Screen * sceneRatio * fov
        y_camera = y_Screen * fov
        z_camera = np.ones((self.h, self.w))

        # make pixels with each combination of x and y values and add z direcotion one unit away
        cameraPixels = np.stack([x_camera, y_camera, z_camera],axis=2)
        cameraPixels = cameraPixels.reshape((cameraPixels.shape[0] * cameraPixels.shape[1], 3))

        # create camera rays from the camera pixel and camera center which is at 0, 0, 0
        cameraCenter = np.array([0, 0, 0])
        cameraRays = cameraPixels - cameraCenter

        # normalize cameraRays
        cameraRays = cameraRays / np.linalg.norm(cameraRays, ord=2, axis=1, keepdims=True)

        # add an extra 1 column to the pixels for transformation
        ones = np.ones((self.h*self.w, 1))
        cameraRays = np.append(cameraRays, ones, axis=1)

        # calculate x, y, z of the camera 
        z_c = np.array([normalize(self.at - self.eye)])

        x_c = np.cross(self.up, z_c) 
        x_c = x_c / np.linalg.norm(x_c, ord=2, axis=1, keepdims=True)

        y_c = np.cross(z_c, x_c)
        y_c = y_c / np.linalg.norm(y_c, ord=2, axis=1, keepdims=True)

        x_c = np.append(x_c, [[0.]], axis=1)
        y_c = np.append(y_c, [[0.]], axis=1)
        z_c = np.append(z_c, [[0.]], axis=1)
        eye = np.append(np.array([self.eye]), [[1.]], axis=1)

        # create camera to world transformation matrix
        A = np.concatenate((x_c, y_c, z_c, eye))

        # transform the pixels from camera space to world space
        world_directions = (cameraRays @ A)[:,:3]

        # get direction to the eye in world space
        world_directions = world_directions - self.eye[np.newaxis, :]

        # normalize the ray direction vectors
        world_directions = world_directions / np.linalg.norm(world_directions, ord=2, axis=1, keepdims=True)

        # create rays with eye at origins and directions vectors
        rays = Rays(np.array([self.eye]), world_directions)

        return rays
        ### END CODE

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """

        ### BEGIN CODE
        hit_ids = np.full(self.w * self.h, -1)
        hit_distances = np.full(self.w * self.h, np.inf)
        hit_normals = np.full((self.w * self.h, 3), np.array([np.inf, np.inf, np.inf]))

        for i in range(len(self.geometries)):
          # intersect rays with geometires
          distances, tempNormals = self.geometries[i].intersect(rays)

          # distance arrays reshaped to use for normals np.where
          normalDist = np.tile(np.array([distances]).transpose(), (3))
          normalMinDist = np.tile(np.array([hit_distances]).transpose(), (3))

          hit_normals = np.where(normalDist < normalMinDist, tempNormals, hit_normals)
          hit_ids = np.where(distances < hit_distances, i, hit_ids)
          hit_distances = np.where(distances < hit_distances, distances, hit_distances)
        ### END CODE

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))
        

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),np.array([0,0,0,1])[np.newaxis,:]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),np.array([0,0,0])[np.newaxis,:]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),np.array([-1])))
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]),np.array([-1])))[ids]


        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and( L_e != np.array([0, 0, 0]), (ids != -1)[:,np.newaxis] ), L_e, L)

        ### BEGIN SOLUTION
        # PLACEHOLDER: our base code renders out debug normals.
        # [TODO] Replace these next three lines with your 
        # solution for your deliverables

        fourth_brdf = brdf_params[:,3]
        brdfs = np.repeat(fourth_brdf[:, np.newaxis], 3, axis=1)

        if(sampling_type == UNIFORM_SAMPLING):    

            ids = np.repeat(ids[:, np.newaxis], 3, axis=1)

            for i in range(len(scene.lights)):
                r1 = np.random.rand(normals.shape[0])
                r2 = np.random.rand(normals.shape[0])

                w_z = 2 * r1 - 1
                r = np.sqrt(1 - np.square(w_z))
                phi = 2 * np.pi * r2
                w_x = r * np.cos(phi) 
                w_y = r * np.sin(phi)
                w = np.array([w_x, w_y, w_z]).T

                shadow_rays = Rays(hit_points + shadow_ray_o_offset * 10 * w, w)

                V_dist, V_normals, V_id = self.intersect(shadow_rays)

                # brdf calculations
                # w_o = np.negative(eye_rays.Ds)
                # dot = np.sum(normals * w_o, axis=1)
                # dot = np.repeat(dot[:, np.newaxis], 3, axis=1)
                # w_r = 2 * (dot * normals) - w_o
                # maximum = np.maximum(0, np.exp(np.sum(w_r * w_o, axis=1), fourth_brdf))
                # phong_multiplier = (brdf_params[:,3] + 1) / (2 * np.pi) * maximum
                # phong_multiplier = np.repeat(phong_multiplier[:, np.newaxis], 3, axis=1)
                # brdf_temp = np.where(brdfs == 1, brdf_params[:,:3] / np.pi, brdf_params[:,:3] * phong_multiplier)

                visibility = np.where(V_id != -1, 0, 1)

                samples = (normals * w).sum(axis=1)
                samples = np.clip(samples, 0, np.amax(samples))

                dots = samples * visibility
                
                light = np.array([scene.lights[i].Le])
                light_val = np.repeat(light, dots.shape[0], axis=0)

                illumination = dots[:, np.newaxis] * light_val 

                L = L + illumination
                # L *= brdf_temp
                

            brdf_temp = np.where(brdfs == 1, brdf_params[:,:3] / np.pi, brdf_params[:,:3] * (brdf_params[:,3] + 1)[:,np.newaxis] / (2 * np.pi))

            # L = L * brdf_temp
            
            L = np.abs(L)
            L = np.clip(L, 0, 1)
            maxVal = np.max(L)
            L = L / maxVal

            # L = np.abs(normals)
            L = L.reshape((self.h, self.w, 3))
            return L
        elif(sampling_type == LIGHT_SAMPLING):
            ids = np.repeat(ids[:, np.newaxis], 3, axis=1)
            alpha = brdf_params[:,3]
            brdfs = brdf_params[:,:3]

            for i in range(len(scene.lights)):
                light = scene.lights[i]
                lightId = -1
                for i in range(len(self.geometries)):
                    if self.geometries[i] == light:
                        lightId = i
                        break

                x = hit_points

                r1 = np.random.rand(x.shape[0])
                r2 = np.random.rand(x.shape[0])

                theta_max = np.arcsin(light.r) / np.linalg.norm(light.c - x, axis=1)

                w_z = 1 - r1 * (1 - np.cos(theta_max))
                r = np.sqrt(1 - np.square(w_z))
                phi = 2 * np.pi * r2
                w_x = r * np.cos(phi)
                w_y = r * np.sin(phi)
                w_x = w_x.reshape(x.shape[0],1)
                w_y = w_y.reshape(x.shape[0],1)
                w_z = w_z.reshape(x.shape[0],1)

                w = np.concatenate((w_x, w_y, w_z), axis=1)

                towards_light = light.c - x

                w_c = (towards_light) / np.linalg.norm(towards_light, axis=1)[:,np.newaxis]
                zeros = np.zeros(x.shape[0])
                option1 = np.array([-w_c[:,2], zeros, w_c[:,0]]).T 
                option1 = option1 / np.linalg.norm(option1, axis=1)[:,np.newaxis]
                option2 = np.array([zeros, w_c[:,2], -w_c[:,1]]).T
                option2 = option2 / np.linalg.norm(option2, axis=1)[:,np.newaxis]
                orth1 = np.where(np.abs(w_c[:,0] > np.abs(w_c[:,1]))[:,np.newaxis], option1, option2)
                orth2 = np.cross(w_c, orth1)
                orth2 = orth2 / np.linalg.norm(orth2, axis=1)[:,np.newaxis]

                worldTransform = np.concatenate((orth1, orth2, w_c), axis=1)
                worldTransform = worldTransform.reshape(x.shape[0], 3, 3)

                w = w[:,np.newaxis,:]
                w_j = w[:] @ worldTransform
                
                w_j = w_j.reshape(x.shape[0], 3)
                w_j = w_j / (2 * np.pi * (1-np.cos(theta_max)[:,np.newaxis]))
                w_j = w_j / np.linalg.norm(w_j, axis=1)[:,np.newaxis]

                shadow_rays = Rays(x + shadow_ray_o_offset * normals, w_j)
                V_dist, V_normals, V_id = self.intersect(shadow_rays)

                pdf = 1 / (2 * np.pi * (1 - np.cos(theta_max)))

                maximum_comp = np.sum(normals * w_j, axis=1)
                maximum_comp = np.clip(maximum_comp, 0, np.amax(maximum_comp))

                intensity = light.Le[np.newaxis]
                intensity = np.repeat(intensity, x.shape[0], axis=0)

                illumination = intensity * maximum_comp[:,np.newaxis]  / pdf[:,np.newaxis]

                illumination = np.where(alpha[:,np.newaxis] == 1, illumination * brdfs / np.pi, illumination)

                w_o = np.negative(eye_rays.Ds)
                w_r = 2 * np.sum(normals * w_o, axis=1)[:,np.newaxis] * normals - w_o
                max = np.power(np.sum(w_r * w_j, axis=1), alpha)
                max = np.clip(max, 0, np.amax(max))

                illumination = np.where(alpha[:,np.newaxis] != 1, illumination * brdfs * (alpha[:,np.newaxis]+1) / (2 * np.pi) * max[:,np.newaxis], illumination)

                # L = np.where(V_id[:,np.newaxis] == lightId, L + illumination, L)
                L = np.where(V_id[:,np.newaxis] == lightId, L + illumination, L)


            
            # L = L / (len(scene.lights))
            L = np.abs(L)
            L = np.clip(L, 0, 1)
            maxVal = np.max(L)
            L = L / maxVal

            # L = np.abs(normals)
            L = L.reshape((self.h, self.w, 3))
            return L
        elif(sampling_type == BRDF_SAMPLING):
            ids = np.repeat(ids[:, np.newaxis], 3, axis=1)
            alpha = brdf_params[:,3]
            brdfs = brdf_params[:,:3]

            for i in range(len(scene.lights)):
                light = scene.lights[i]
                lightId = -1
                for i in range(len(self.geometries)):
                    if self.geometries[i] == light:
                        lightId = i
                        break

                x = hit_points

                r1 = np.random.rand(x.shape[0])
                r2 = np.random.rand(x.shape[0])

                

                # diffuse
                r1D = np.where(alpha == 1, r1, 0)
                r2D = np.where(alpha == 1, r2, 0)
                w_z = 2 * r1D - 1
                r = np.sqrt(1 - np.square(w_z))
                phi = 2 * np.pi * r2D
                w_x = r * np.cos(phi) 
                w_y = r * np.sin(phi)
                w = np.array([w_x, w_y, w_z]).T

                shadow_rays = Rays(hit_points + shadow_ray_o_offset * normals, w)

                V_dist, V_normals, V_id = self.intersect(shadow_rays)

                L = np.where(np.logical_and(V_id[:,np.newaxis] == lightId, alpha[:,np.newaxis] == 1), L + (light.Le * brdfs), L)

                r1S = np.where(alpha != 1, r1, 0)
                r2S = np.where(alpha != 1, r2, 0)

                wS_z = r1S**(1/(alpha+1))
                rS = np.sqrt(1 - np.square(wS_z))
                phiS = 2 * np.pi * r2S
                wS_x = rS * np.cos(phiS)
                wS_y = rS * np.sin(phiS)
                wS = np.array([wS_x, wS_y, wS_z]).T

                reflection = eye_rays.Ds - 2 * np.sum(eye_rays.Ds * normals, axis=1)[:,np.newaxis] * normals

                w_c = (reflection) / np.linalg.norm(reflection, axis=1)[:,np.newaxis]
                zeros = np.zeros(x.shape[0])
                option1 = np.array([-w_c[:,2], zeros, w_c[:,0]]).T 
                option1 = option1 / np.linalg.norm(option1, axis=1)[:,np.newaxis]
                option2 = np.array([zeros, w_c[:,2], -w_c[:,1]]).T
                option2 = option2 / np.linalg.norm(option2, axis=1)[:,np.newaxis]
                orth1 = np.where(np.abs(w_c[:,0] > np.abs(w_c[:,1]))[:,np.newaxis], option1, option2)
                orth2 = np.cross(w_c, orth1)
                orth2 = orth2 / np.linalg.norm(orth2, axis=1)[:,np.newaxis]

                worldTransform = np.concatenate((orth1, orth2, w_c), axis=1)
                worldTransform = worldTransform.reshape(x.shape[0], 3, 3)

                w = wS[:,np.newaxis,:]
                w_j = w[:] @ worldTransform

                w_j = w_j.reshape(x.shape[0], 3)
                w_j = w_j / np.linalg.norm(w_j, axis=1)[:,np.newaxis]

                shadow_raysS = Rays(hit_points + shadow_ray_o_offset * normals, w_j)

                V_dist, V_normals, V_idS = self.intersect(shadow_raysS)
                max = np.sum(normals * w_j, axis = 1)
                max = np.clip(max, 0, np.amax(max))

                L = np.where(np.logical_and(V_idS[:,np.newaxis] == lightId, alpha[:,np.newaxis] != 1), L + (light.Le * brdfs), L)

            L = np.abs(L)
            L = np.clip(L, 0, 1)
            maxVal = np.max(L)
            L = L / maxVal

            # L = np.abs(normals)
            L = L.reshape((self.h, self.w, 3))
            return L
        else:
            L = np.abs(normals)
            L = L.reshape((self.h, self.w, 3))
            return L
        ### END SOLUTION

    def progressive_render_display( self, jitter = False, total_spp = 20,
                                    sampling_type = UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the 
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        ### BEGIN CODE (note: we will not grade your progressive rendering code in A3)
        # [TODO] replace the next five lines with 
        # your progressive rendering display loop
        current_average = np.array([0., 0., 0.])
        
        for i in range(total_spp):
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            plt.title(f"current spp: {i+1} of {total_spp}")
            L = self.render(vectorized_eye_rays, sampling_type)
            current_average = current_average + L               
            image_data.set_data(current_average)
            plt.pause(0.001) # add a tiny delay between rendering passes
        ### END CODE

        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)

if __name__ == "__main__":
    enabled_tests = [True, True, False]

    # Create test scene and test sphere
    scene = Scene(w=int(512), h=int(512)) # TODO: debug at lower resolution
    scene.set_camera_parameters(
        eye=np.array([0, 2, 15], dtype=np.float64),
        at=normalize(np.array([0, -2, 2.5], dtype=np.float64)),
        up=np.array([0, 1, 0], dtype=np.float64),
        fov=int(40)
    )

    # Veach Scene Lights
    scene.add_geometries([ Sphere( 0.0333, np.array([3.75, 0, 0]),
                                   Le = 10 * np.array([901.803, 0, 0]) ),
                           Sphere( 0.1, np.array([1.25, 0, 0]),
                                   Le = 10 * np.array([0, 100, 0]) ),
                           Sphere( 0.3, np.array([-1.25, 0, 0]),
                                   Le = 10 * np.array([0, 0, 11.1111]) ),
                           Sphere( 0.9, np.array([-3.75, 0, 0]),
                                   Le = 10 * np.array([1.23457, 1.23457, 1.23457]) ),
                           Sphere( 0.5, np.array([-10, 10, 4]),
                                   Le = np.array([800, 800, 800]) ) ] ) 
                           
    # Geometry
    scene.add_geometries( [ Mesh( "plate1.obj", 
                                   brdf_params = np.array( [1,1,1,30000] ) ),
                            Mesh( "plate2.obj", 
                                   brdf_params = np.array( [1,1,1,5000] ) ),
                            Mesh( "plate3.obj", 
                                   brdf_params = np.array( [1,1,1,1500] ) ),
                            Mesh( "plate4.obj", 
                                   brdf_params = np.array( [1,1,1,100] ) ),
                            Mesh( "floor.obj", 
                                   brdf_params = np.array( [0.5,0.5,0.5,1] ) ) ])

    #########################################################################
    ### Deliverable 1 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[0]:
        scene.progressive_render_display(total_spp = 1024, jitter = True, sampling_type = LIGHT_SAMPLING)
        # scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = UNIFORM_SAMPLING)
        # scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = UNIFORM_SAMPLING)

    #########################################################################
    ### Deliverable 2 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[1]:
        scene.progressive_render_display(total_spp = 1, jitter = True, sampling_type = BRDF_SAMPLING)
        # scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = LIGHT_SAMPLING)
        # scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = LIGHT_SAMPLING)

    #########################################################################
    ### Deliverable 3 TEST (Only for ECSE 546 students!): comment/modify as you see fit
    #########################################################################
    if enabled_tests[2]:  
        scene.progressive_render_display(total_spp = 1, jitter = True, sampling_type = MIS_SAMPLING)
        scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = MIS_SAMPLING)
        scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = MIS_SAMPLING)
        