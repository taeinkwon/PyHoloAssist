import argparse
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import tqdm
import sys
import shutil
from ffprobe import FFProbe

axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


class IndexSearch():
    def __init__(self, time_array):
        self.time_array = time_array
        self.prev = 0
        self.index = 0
        self.len = len(time_array)

    def nearest_neighbor(self, target_time):
        while(target_time > self.time_array[self.index]):
            if self.len - 1 <= self.index:
                return self.index
            self.index += 1
            self.prev = self.time_array[self.index]
        
        if (abs(self.time_array[self.index] - target_time) > abs(self.time_array[self.index-1] - target_time)) and (self.index != 0):
            ret_index = self.index-1
        else:
            ret_index = self.index
        return ret_index


def get_handpose_connectivity():
    # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
    return [
        [0, 1],

        # Thumb
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],

        # Index
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],

        # Middle
        [1, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Ring
        [1, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        # Pinky
        [1, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [24, 25]
    ]


def read_pose_txt(img_pose_path):
    img_pose_array = []
    with open(img_pose_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            # pose = np.array(line_data[1:]).reshape(4, 4)
            # pose = np.dot(axis_transform,pose)
            # line_data[1:] = pose.reshape(-1)
            # print("line_data",line_data)
            # line_data = line.strip().split('\t')
            img_pose_array.append(line_data)
        img_pose_array = np.array(img_pose_array)
    return img_pose_array


def read_hand_pose_txt(hand_path):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            line_data = list(map(float, line.split('\t')))
            line_data_reshape = np.reshape(
                line_data[3:-52], (-1, 4, 4))  # For version2: line_data[5:-52]

            line_data_xyz = []
            for line_data_reshape_elem in line_data_reshape:
                # To get translation of the hand joints
                location = np.dot(line_data_reshape_elem,
                                np.array([[0, 0, 0, 1]]).T)
                line_data_xyz.append(location[:3].T[0])

            line_data_xyz = np.array(line_data_xyz).T
            hand = line_data[:4]
            hand.extend(line_data_xyz.reshape(-1))
            hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array


def read_intrinsics_txt(img_instrics_path):
    with open(img_instrics_path) as f:
        data = list(map(float, f.read().split('\t')))
        intrinsics = np.array(data[:9]).reshape(3, 3)
        width = data[-2]
        height = data[-1]
    return intrinsics, width, height


def read_gaze_txt(gaze_path):
    with open(gaze_path) as f:
        gaze_data = []
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            gaze_data.append(line_data)
        gaze_data = np.array(gaze_data)
    return gaze_data


def get_eye_gaze_point(gaze_data, dist):
    origin_homog = gaze_data[2:5]
    direction_homog = gaze_data[5:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    point = origin_homog + direction_homog * dist

    return point[:3]

def get_framerate(filename):
    metadata=FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            #print("stream",stream.__dir__())
            print('Stream contains {} frames.'.format(stream.framerate))
            framerate = float(stream.framerate)
            frame_num = int(stream.frames())
            
    return framerate, frame_num



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/media/taein/8tb_sdd/datasets/HoloAssist/HoloAssist',
                        help='Directory of untar data')
    parser.add_argument('--video_name', type=str, default='z013-june-16-22-gopro', help="Directory sequence")
    parser.add_argument('--frame_num', type=int, default=None,
                        help='Specific number of frame')
    parser.add_argument('--eye', action='store_true',
                        help='Proejct eye gaze on images')
    parser.add_argument('--save_eyeproj', action='store_true',
                        help='Save eyeproj.txt file.')
    parser.add_argument('--save_video', action='store_true',
                        help='Save hand_project_mpeg file.')
    parser.add_argument('--eye_dist', type=float, default=0.5,
                        help='Eyegaze projection dist is 50cm by default')
    parser.add_argument('--offset', type=int, default=0,
                        help='Temporal offset for image and hand poses (ms), hand pose processing time is approximately 50ms')
    args = parser.parse_args()
    
    base_path = os.path.join(args.folder_path, args.video_name, "Export_py")

    if not os.path.exists(base_path):
        # Exit if the path does not exist
        sys.exit('{} does not exist'.format(base_path))
        
    mpeg_img_path = os.path.join(base_path,"Video","images")
    mpeg_aligned_img_path = os.path.join(base_path,"Video","images_aligned")
    hands_path = os.path.join(base_path, 'Hands')
    img_path = os.path.join(base_path, 'Video')
    

    # Read timing file
    img_sync_timing_path = os.path.join(img_path, 'Pose_sync.txt')
    img_sync_timing_array = []
    with open(img_sync_timing_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = int(line.split('\t')[1])
            img_sync_timing_array.append(line_data)
    
    start_time_path = os.path.join(img_path, 'VideoMp4Timing.txt')
    with open(start_time_path) as f:
        lines = f.read().split('\n')
        start_time = int(lines[0])
        end_time = int(lines[1])
        

    frame_rate, frame_num = get_framerate(os.path.join(base_path,"Video_pitchshift.mp4"))
    img_timing_array = []
    for ii in range(frame_num):
        img_timing_array.append(int(start_time + ii * (1/frame_rate)* 10**7))
        
    
    
    #print("img_timing_array",img_timing_array[-1] - end_time)
    
    if args.save_video:
        os.chdir(os.path.join(base_path,"Video"))
        if not os.path.exists(mpeg_img_path):
            # Export images if the path does not exist.
            os.mkdir(mpeg_img_path)
            os.system("ffmpeg -i ../Video_pitchshift.mp4 -start_number 0 images/%06d.png")
        

        if not os.path.exists(mpeg_aligned_img_path):
            os.mkdir(mpeg_aligned_img_path)
            timestamp_array_mepg = np.array(img_timing_array)
            mpeg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
                    timestamp_array_mepg.reshape(-1, 1))
        
            for ii , img_sync_timestamp in enumerate(img_sync_timing_array[:]):
                _, mpeg_indices = mpeg_nbrs.kneighbors(
                    np.array(img_sync_timestamp).reshape(-1, 1))
                source = os.path.join(mpeg_img_path,'{0:06d}.png'.format(mpeg_indices[0][0]))
                dest = os.path.join(mpeg_aligned_img_path,'{0:06d}.png'.format(ii))
                shutil.copy(source , dest)
            
    
    # Read left hand
    left_hand_path = os.path.join(hands_path, 'Left_sync.txt')
    left_hand_array = read_hand_pose_txt(left_hand_path)
    # Read right hand
    right_hand_path = os.path.join(hands_path, 'Right_sync.txt')
    right_hand_array = read_hand_pose_txt(right_hand_path)

    left_hand_pose_timestamp = left_hand_array[:, 1]  # .astype(np.float64)
    right_hand_pose_timestamp = right_hand_array[:, 1]  # .astype(np.float64)

    # Read gaze
    if args.eye:
        gaze_path = os.path.join(base_path, "Eyes", "Eyes_sync.txt")
        gaze_array = read_gaze_txt(gaze_path)
        gaze_timestamp = gaze_array[:, :2]
        eyeproj_list = []

    
    # Project into the image
    projected_path = os.path.join(base_path, "projected_mpeg_img")
    if not os.path.exists(projected_path):
        # Create a new directory because it does not exist
        os.makedirs(projected_path)
    # Video setup
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
        video_out = cv2.VideoWriter(os.path.join(base_path, 'project.mp4'), fourcc, frame_rate, (896, 504), isColor=True)

    num_frames = len(img_sync_timing_array)
    # Read campose
    img_pose_path = os.path.join(img_path, 'Pose_sync.txt')
    img_pose_array = read_pose_txt(img_pose_path)
    # Read cam instrics
    img_instrics_path = os.path.join(img_path, 'Intrinsics.txt')
    img_intrinsics, width, height = read_intrinsics_txt(img_instrics_path)
    

    for frame in tqdm.tqdm(range(num_frames)):
        if args.frame_num != None and frame != args.frame_num:
            continue

             
        left_indices = [[frame]]
        right_indices = [[frame]]
       
        hand_point_left = left_hand_array[left_indices[0][0]][4:].reshape(
            3, -1)
        hand_point_right = right_hand_array[right_indices[0][0]][4:].reshape(
            3, -1)

        img_pose = img_pose_array[frame][2:].reshape(4, 4)

        ''' calculate extrinsics first and apply coordinate system transformation
        '''
        
        # hand pose to the camera coordinate.
        hand_point_right = np.dot(axis_transform, np.dot(np.linalg.inv(
            img_pose), np.concatenate((hand_point_right, [[1]*np.shape(hand_point_right)[1]]))))
        hand_point_left = np.dot(axis_transform, np.dot(np.linalg.inv(
            img_pose), np.concatenate((hand_point_left, [[1]*np.shape(hand_point_right)[1]]))))
        
        # Put an empty camera pose for image.
        rvec = np.array([[0.0, 0.0, 0.0]])
        tvec = np.array([0.0, 0.0, 0.0])

        

        # For eyes
        if args.eye:
            gaze_indices = left_indices = [[frame]]
            point = get_eye_gaze_point(gaze_array[gaze_indices[0][0]], args.eye_dist)
            
            point_transformed = np.dot(axis_transform, np.dot(np.linalg.inv(
                img_pose), np.concatenate((point, [1]))))

            img_points_gaze, _ = cv2.projectPoints(
                point_transformed[:3].reshape((1, 3)), rvec, tvec, img_intrinsics, np.array([]))
            eyeproj_list.append(img_points_gaze[0][0])
            
        if args.save_video:
            # Blue color in BGR
            img = cv2.imread(os.path.join(mpeg_aligned_img_path, '{0:06d}.png'.format(frame)))   
            img_points_left, _ = cv2.projectPoints(
                hand_point_left[:3], rvec, tvec, img_intrinsics, np.array([]))
            img_points_right, _ = cv2.projectPoints(
                hand_point_right[:3], rvec, tvec, img_intrinsics, np.array([]))
            connectivity = get_handpose_connectivity()
            color = (255, 0, 0)
            radius = 5
            thickness = 2

            points = img_points_left
            #print("points",points)
            if not (np.isnan(points).any()):
                for limb in connectivity:
                    cv2.line(img, (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
                            (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])), color, thickness)
            color = (0, 255, 0)

            points = img_points_right
            if not (np.isnan(points).any()):
                for limb in connectivity:
                    cv2.line(img, (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
                            (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])), color, thickness)

            if args.eye:
                
                points = img_points_gaze
                if not np.isnan(points[0][0]).any():
                    color = (0, 0, 255)
                    thickness = 4
                    radius = 10
                    cv2.circle(img, (int(points[0][0][0]), int(
                        points[0][0][1])), radius, color, thickness)
                #filename = os.path.join(projected_path, '{0:06d}.png'.format(frame))

            video_out.write(img)
    if args.save_video:
        cv2.destroyAllWindows()
        video_out.release()
    if args.eye and args.save_eyeproj:
        with open(os.path.join(base_path, "Eyes",'Eyes_proj.txt'), 'w') as f:
            for ii, elems in enumerate(eyeproj_list):
                #print("gaze_timestamp",np.shape(gaze_timestamp))
                
                f.write(f"{gaze_timestamp[ii,0]}\t {gaze_timestamp[ii,1]}\t")
                for elem in elems:
                    f.write(f"{elem}\t")
                f.write("\n")

if __name__ == '__main__':
    main()
