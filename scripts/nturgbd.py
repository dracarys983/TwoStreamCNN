import os
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn import preprocessing

import tensorflow as tf
from tensorflow import logging
logging.set_verbosity(tf.logging.INFO)

class Joint(object):
    def __init__(self,
                 x, y, z,
                 dX=None, dY=None,
                 cX=None, cY=None,
                 orX=None, orY=None, orZ=None, orW=None,
                 tState=None):
        self.x = x; self.y = y; self.z = z
        self.depthX = dX; self.depthY = dY
        self.colorX = cX; self.colorY = cY
        self.orientX = orX; self.orientY = orY
        self.orientZ = orZ; self.orientW = orW
        self.tracker = tState

    def _calculate_cylindrical_coordinates(self):
        rho = np.sqrt(self.x**2 + self.y**2)
        if self.x == 0 and self.y == 0:
            phi = 0
        elif self.x >= 0:
            phi = np.arcsin(self.y/rho)
        elif self.x > 0:
            phi = np.arctan2(self.y, self.x)
        elif self.x < 0:
            phi = -np.arcsin(self.y/rho) + np.pi
        return rho, phi, self.z

    def _calculate_spherical_coordinates(self):
        xy = self.x**2 + self.y**2
        r = np.sqrt(xy + self.z**2)
        theta = np.arctan2(self.z, xy)
        if self.x == 0 and self.y == 0:
            phi = 0
        elif self.x >= 0:
            phi = np.arcsin(self.y/np.sqrt(xy))
        elif self.x > 0:
            phi = np.arctan2(self.y, self.x)
        elif self.x < 0:
            phi = -np.arcsin(self.y/np.sqrt(xy)) + np.pi
        return r, theta, phi

    def _get_cartesian_coordinates(self):
        return self.x, self.y, self.z

    def _set_cartesian_coordinates(self, x, y, z):
        self.x = x; self.y = y; self.z = z

    def _get_cylindrical_coordinates(self):
        return self._calculate_cylindrical_coordinates()

    def _get_spherical_coordinates(self):
        return self._calculate_spherical_coordinates()

    def _get_depth_coordinates(self):
        return self.depthX, self.depthY

    def _get_rgb_coordinates(self):
        return self.colorX, self.colorY

    def _get_orientation_coordinates(self):
        return self.orientX, self.orientY, self.orientZ, self.orientW

class Skeleton(object):
    def __init__(self,
                 skelid=None,
                 njoints=None,
                 clip=None,
                 lconfidence=None, lstate=None,
                 rconfidence=None, rstate=None,
                 restrict=None,
                 lX=None, lY=None,
                 tracker=None,
                 joints=None):
        self.skeletonID = skelid
        self.num_joints = njoints
        if joints is None:
            self.joints = []
        else:
            self.joints = joints
        self.clip_edges = clip
        self.left_hand_confidence = lconfidence
        self.left_hand_state = lstate
        self.right_hand_confidence = rconfidence
        self.right_hand_state = rstate
        self.tracker = tracker
        self._is_zero_skeleton = False

    def _get_skeleton_id(self):
        return self.skeletonID

    def _get_num_joints(self):
        return self.num_joints

    def _get_joint_objects(self):
        return self.joints

    def _get_clip_edges_val(self):
        return self.clip_edges

    def _get_left_hand(self):
        return self.left_hand_confidence, self.left_hand_state

    def _get_right_hand(self):
        return self.right_hand_confidence, self.right_hand_state

    def _set_joint_objects(self, joints):
        self.joints = joints

    def _add_joint_object(self, joint):
        self.joints.append(joint)


class Frame(object):
    def __init__(self,
                 nskels=None,
                 skels=None):
        self.num_skeletons = nskels
        if skels is None:
            self.skeletons = []

    def _get_num_skeletons(self):
        return self.num_skeletons

    def _get_skeleton_objects(self):
        return self.skeletons

    def _set_skeleton_objects(self, skeletons):
        self.skeletons = skeletons

    def _add_skeleton_object(self, skeleton):
        self.skeletons.append(skeleton)

class SkeletonVideo(object):
    def __init__(self,
                 nframes,
                 frames=None):
        self.num_frames = nframes
        if frames is None:
            self.frames = []

    def _get_num_frames(self):
        return self.num_frames

    def _get_frame_objects(self):
        return self.frames

    def _set_frame_objects(self, frames):
        self.frames = frames

    def _add_frame_object(self, frame):
        self.frames.append(frame)

    def _get_main_actor_skeletons(self):

        def _get_motion_for_skeletons(skeletons):
            total_dist = 0
            for i in range(len(skeletons)-1):
                joints_1 = skeletons[i]._get_joint_objects()
                joints_2 = skeletons[i+1]._get_joint_objects()
                for j, k in zip(joints_1, joints_2):
                    p1 = np.array(j._get_cartesian_coordinates())
                    p2 = np.array(k._get_cartesian_coordinates())
                    dist = np.sqrt(np.sum(np.square(p1 - p2)))
                    total_dist += dist
            return total_dist

        def _is_noisy_skeleton(skeleton):
            joints = skeleton._get_joint_objects()
            X = []; Y = []
            for joint in joints:
                x, y, z = joint._get_cartesian_coordinates()
                X.append(x); Y.append(y)
            X = np.array(X); Y = np.array(Y)
            xspread = np.max(X) - np.min(X); yspread = np.max(Y) - np.min(Y)
            return (yspread / xspread)

        def _create_zero_skeleton():
            joints = []
            for i in range(25):
                joint = Joint(0.0, 0.0, 0.0)
                joints.append(joint)
            skeleton = Skeleton(njoints=len(joints), joints=joints)
            skeleton._is_zero_skeleton = True
            return skeleton

        skeletons_0 = []; skeletons_1 = []
        for i in range(len(self.frames)):
            frame = self.frames[i]
            if frame._get_num_skeletons() == 1:
                skeletons_0.append(frame._get_skeleton_objects()[0])
                skeletons_1.append(_create_zero_skeleton())
            elif frame._get_num_skeletons() == 2:
                skeletons_0.append(frame._get_skeleton_objects()[0])
                skeletons_1.append(frame._get_skeleton_objects()[1])
            elif frame._get_num_skeletons() > 2:
                ratios = []
                for i in range(frame._get_num_skeletons()):
                    skeleton = frame._get_skeleton_objects()[i]
                    ratios.append((_is_noisy_skeleton(skeleton), i))
                ratios = sorted(ratios)
                idx = [x for (val, x) in ratios[-2:]]
                skeletons_0.append(frame._get_skeleton_objects()[idx[0]])
                skeletons_1.append(frame._get_skeleton_objects()[idx[1]])
        dist_0 = _get_motion_for_skeletons(skeletons_0)
        dist_1 = _get_motion_for_skeletons(skeletons_1)
        if dist_0 > dist_1:
            return skeletons_0, skeletons_1
        return skeletons_1, skeletons_0


# Reads the data for a complete frame set from the NTU RGB+D Action Recognition Dataset
# Included joints are:
# --------------------------------------------------------------------------------------------------------------
# 	0 -  base of the spine
# 	1 -  middle of the spine
# 	2 -  neck
#	3 -  head
# 	4 -  left shoulder
# 	5 -  left elbow
# 	6 -  left wrist
# 	7 -  left hand
# 	8 -  right shoulder
# 	9 - right elbow
# 	10 - right wrist
# 	11 - right hand
# 	12 - left hip
# 	13 - left knee
# 	14 - left ankle
# 	15 - left foot
# 	16 - right hip
# 	17 - right knee
# 	18 - right ankle
# 	19 - right foot
# 	20 - spine
#	21 - tip of the left hand
# 	22 - left thumb
# 	23 - tip of the right hand
# 	24 - right thumb
# --------------------------------------------------------------------------------------------------------------
class Reader(object):
    def __init__(self,
                 dataset_dir,
                 splits_dir):
        self.data = dataset_dir
        self.splits = splits_dir

        self.train_splits = {1: os.path.join(splits_dir, 'train_cs.txt'),
                2: os.path.join(splits_dir, 'train_cv.txt')}
        self.test_splits = {1: os.path.join(splits_dir, 'test_cs.txt'),
                2: os.path.join(splits_dir, 'test_cv.txt')}

    def _normalize_skeleton(self, skeleton):
        joints = skeleton._get_joint_objects()
        if not (len(joints) == 25):
            return None

        ''' Translation Matrix
        - T_x: neg(X coordinate of middle of spine)
        - T_y: neg(Y coordinate of middle of spine)
        - T_z: neg(Z coordinate of middle of spine)
        '''
        origin = joints[1]
        transmat = np.zeros((4, 4))
        transmat[0][0] = transmat[1][1] = transmat[2][2] = transmat[3][3] = 1.0
        transmat[3][0] = -origin.x; transmat[3][1] = -origin.y; transmat[3][2] = -origin.z

        ''' Rotation Matrix
        - New X axis: Right shoulder (rs) to Left shoulder (ls) joint
        - New Y axis: Base of spine (bsp) to Spine (sp) joint
        - New Z axis: In direction  of X cross Y
        - Using arctan2 is always better than using arcsin/arccos, as they can be
        numerically unstable for certain values of the angles.
        '''
        rs = joints[8]; ls = joints[4]; bsp = joints[0]; sp = joints[20]
        rs = np.array(rs._get_cartesian_coordinates())
        ls = np.array(ls._get_cartesian_coordinates())
        bsp = np.array(bsp._get_cartesian_coordinates())
        sp = np.array(sp._get_cartesian_coordinates())
        curr_x = np.array([1.0, 0.0, 0.0]); new_x = np.add(rs, -ls)
        curr_y = np.array([0.0, 1.0, 0.0]); new_y = np.add(bsp, -sp)
        curr_z = np.array([0.0, 0.0, 1.0]); new_z = np.cross(new_x, new_y)
        # Dot and cross product both needed for arctan2
        x_dot = np.dot(new_x, curr_x); y_dot = np.dot(new_y, curr_y); z_dot = np.dot(new_z, curr_z)
        x_cross = np.cross(new_x, curr_x); y_cross = np.cross(new_y, curr_y); z_cross = np.cross(new_z, curr_z)
        # arccos is numerically unstable when angles are near zero
        theta_x = np.arctan2(np.linalg.norm(x_cross), x_dot)
        theta_y = np.arctan2(np.linalg.norm(y_cross), y_dot)
        theta_z = np.arctan2(np.linalg.norm(z_cross), z_dot)
        rot_x = np.zeros((4, 4)); rot_x[0][0] = 1.0; rot_x[3][3] = 1.0
        rot_x[1][1] = np.cos(theta_x); rot_x[1][2] = -np.sin(theta_x)
        rot_x[2][1] = np.sin(theta_x); rot_x[2][2] = np.cos(theta_x)
        rot_y = np.zeros((4, 4)); rot_y[1][1] = 1.0; rot_y[3][3] = 1.0
        rot_y[0][0] = np.cos(theta_y); rot_y[0][2] = np.sin(theta_y)
        rot_y[2][0] = -np.sin(theta_y); rot_y[2][2] = np.cos(theta_y)
        rot_z = np.zeros((4, 4)); rot_z[2][2] = 1.0; rot_z[3][3] = 1.0
        rot_z[0][0] = np.cos(theta_z); rot_z[0][1] = -np.sin(theta_z)
        rot_z[1][0] = np.sin(theta_z); rot_z[1][1] = np.cos(theta_z)

        ''' Scaling Matrix
        - S_x, S_y, S_z: Inverse of distance between Base of spine (bsp) and
        spine (sp) joint; add a small delta to avoid division by zero (in case
        of very small values of distance).
        '''
        scale = np.zeros((4, 4)); scale[3][3] = 1.0
        dist = np.linalg.norm(new_y) + 1e-4
        scale[0][0] = 1.0/(dist); scale[1][1] = 1.0/(dist); scale[2][2] = 1.0/(dist)

        new_joints = []
        for joint in joints:
            x, y, z = joint._get_cartesian_coordinates()
            # (4,) : Homogeneous coordinates
            j = np.array([x, y, z, 1.0])
            # (4,) x (4, 4) = (4,) for all matrix multiplications
            j = np.matmul(j, transmat)
            j = np.matmul(j, rot_x); j = np.matmul(j, rot_y); j = np.matmul(j, rot_z)
            j = np.matmul(j, scale)
            # Get (x, y, z) from Homogeneous coordinates
            joint._set_cartesian_coordinates(j[0], j[1], j[2])
            new_joints.append(joint)
        skeleton._set_joint_objects(new_joints)

        return skeleton

    def _read_skeleton_file(self, f):
        fpath = os.path.join(self.data, f)
        with open(fpath, 'r') as f:
            out = f.read().replace('\n', '').split()

        index = 0
        nframes = int(out[index]); index += 1
        video = SkeletonVideo(nframes)
        for i in range(nframes):
            bods = int(out[index]); index += 1
            frame = Frame()
            for j in range(bods):
                skelid = long(out[index]); index += 1

                cedges = int(out[index]); index += 1
                lconfidence = int(out[index]); index += 1
                lstate = int(out[index]); index += 1
                rconfidence = int(out[index]); index += 1
                rstate = int(out[index]); index += 1
                restrict = int(out[index]); index += 1

                lX = float(out[index]); index += 1
                lY = float(out[index]); index += 1

                track = int(out[index]); index += 1

                num_joints = int(out[index]); index += 1
                skeleton = Skeleton(skelid, num_joints,
                                    cedges, lconfidence, lstate,
                                    rconfidence, rstate, restrict,
                                    lX, lY, track)

                for k in range(num_joints):
                    x = float(out[index]); index += 1
                    y = float(out[index]); index += 1
                    z = float(out[index]); index += 1

                    dX = float(out[index]); index += 1
                    dY = float(out[index]); index += 1

                    cX = float(out[index]); index += 1
                    cY = float(out[index]); index += 1

                    orW = float(out[index]); index += 1
                    orX = float(out[index]); index += 1
                    orY = float(out[index]); index += 1
                    orZ = float(out[index]); index += 1

                    track = int(out[index]); index += 1

                    joint = Joint(x, y, z,
                                  dX, dY,
                                  cX, cY,
                                  orX, orY, orZ, orW,
                                  track)
                    skeleton._add_joint_object(joint)
                skeleton = self._normalize_skeleton(skeleton)
                if skeleton:
                    frame._add_skeleton_object(skeleton)
                frame.num_skeletons = len(frame._get_skeleton_objects())
            video._add_frame_object(frame)
        return video

    def _generate_image_representation_no_features(self, skeleton):
        joints = skeleton._get_joint_objects()
        xlist = []; ylist = []; zlist = []
        for joint in joints:
            x, y, z = joint._get_cartesian_coordinates()
            xlist.append(x); ylist.append(y); zlist.append(z)
        jointlist = [[0, 1, 2, 3], [0, 16, 17, 18, 19], [0, 12, 13, 14, 15],
                [20, 4, 5, 6, 7, 21], [20, 8, 9, 10, 11, 23], [11, 24], [7, 22]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(jointlist)):
            x_plot = []; y_plot = []; z_plot = []
            for j in jointlist[i]:
                x_plot.append(xlist[j]); y_plot.append(ylist[j]); z_plot.append(zlist[j])
            ax.scatter(x_plot, y_plot, z_plot, c = 'b')
            ax.plot(x_plot, y_plot, z_plot, c = 'b')
        plt.show()
