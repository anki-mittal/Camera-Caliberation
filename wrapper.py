import cv2 as cv
import numpy as np
import os
from typing import List
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def read_caliberation_images(folder_path):
    """
    Reads all images from the specified folder using OpenCV.

    :param folder_path: Path to the folder containing images.
    :return: A list of images in OpenCV format.
    """
    images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add or remove formats as needed

    for filename in os.listdir(folder_path):
        if filename.endswith(supported_formats):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv.imread(img_path)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Failed to load image: {filename}")
            except IOError as e:
                print(f"Error in loading image: {filename}. Error: {e}")

    return images

def getcheckerboradconers(images,chessboardSize):
    imgpoints = []
    count = 1
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret == True:

            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # print(corners)
            # sdf
            # Draw and display the corners
            cv.drawChessboardCorners(image, chessboardSize, corners2, ret)
            # cv.imshow('img', image)
            # cv.imwrite("Results/" + str(count) + ".png", image)
            # plt.imshow(image)
            # plt.show()
            count = count+1
            # cv.waitKey(0)
            # dsf

    # cv.destroyAllWindows()
    return imgpoints

def find_homography(img_corners, worldpoints):
    if len(img_corners) != len(worldpoints) or len(img_corners) < 4:
        raise ValueError("There must be at least 4 correspondences and the number of image corners must match the number of world points.")
    
    h = []
    for i in range(len(img_corners)):
        img_corners = np.squeeze(img_corners)
        # print(worldpoints)
        xi, yi = img_corners[i]
        Xi, Yi = worldpoints[i]
        ax_T = np.array([-Xi, -Yi, -1, 0, 0, 0, xi*Xi, xi*Yi, xi])
        h.append(ax_T)
        ay_T = np.array([0, 0, 0, -Xi, -Yi, -1, yi*Xi, yi*Yi, yi])
        h.append(ay_T)
    
    h = np.array(h)
    _, _, V_T = np.linalg.svd(h, full_matrices=True)
    H = V_T[-1].reshape(3, 3)  # Reshape the last row of V_T to a 3x3 matrix
    # Normalize the homography matrix to ensure H[2, 2] is 1
    H = H / H[2, 2]
    # print(H)
    return H

def calculate_homography(imgpoints, worldpoints):
    Homography = []
    for img_corners in imgpoints:
        H = find_homography(img_corners, worldpoints)
        Homography.append(H)
    Homography = np.array(Homography)
    return Homography

def Get_V_ij(H, i, j):
    H = H.T
    i -=1
    j -=1
    return np.array([[H[i][0]*H[j][0]],
                  [H[i][0]*H[j][1] + H[i][1]*H[j][0]],
                  [H[i][1]*H[j][1]],
                  [H[i][2]*H[j][0] + H[i][0]*H[j][2]],
                  [H[i][2]*H[j][1] + H[i][1]*H[j][2]],
                  [H[i][2]*H[j][2]]]).T


def get_intrinsic_mat(b_vec):
    """
    Calculates the intrinsic matrix from the b vector according to Appendix A
    in the reference paper.
    """
    b11 = b_vec[0]
    b12 = b_vec[1]
    b22 = b_vec[2]
    b13 = b_vec[3]
    b23 = b_vec[4]
    b33 = b_vec[5]
    v = ((b12 * b13) - (b11 * b23)) / ((b11 * b22) - b12 ** 2)
    lmda = b33 - ((b13 ** 2) + (v * (b12 * b13 - b11 * b23))) / b11
    alpha = np.sqrt(lmda / b11)
    beta = np.sqrt(lmda * b11 / ((b11 * b22) - (b12 ** 2)))
    gamma = -b12 * (alpha ** 2) * beta / lmda
    u = gamma * v / beta - b13 * (alpha ** 2) / lmda

    return np.array([
        [alpha, gamma, u],
        [0,     beta,  v],
        [0,     0,     1]
    ])

def getmatrix_B(homography):
    v = []
    for i in range (len(homography)):
        H = homography[i]
        v.append(Get_V_ij(H, 1,2))
        v.append((Get_V_ij(H, 1, 1) - Get_V_ij(H, 2, 2)))
    v = np.array(v)
    v=  np.squeeze(v)
    U, S, V_T = np.linalg.svd(v)    #decomposed b into three matrices                 
    b = V_T[-1]
    return b

def get_exterinsic_mat(K,homography):
    A_inv = np.linalg.pinv(K)
    Rt = []
    for h in homography:
        h1 = h[:,0]
        h2 = h[:,1]
        h3 = h[:,2]
        r1 = A_inv@h1
        r2 = A_inv@h2
        lamda1 = 1/np.linalg.norm(r1)
        lamda2 = 1/np.linalg.norm(r2)

        r1= lamda1*r1
        r2= lamda2*r2
        r3 = np.cross(r1,r2)
        t = lamda1*(A_inv@h3)
        R = np.vstack((r1, r2, r3, t)).T
        Rt.append(R)
    return Rt

def package_x_vector(K, k_distortion):
    return np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], k_distortion[0], k_distortion[1]])

def convert_xvec_to_Amatrix(x0):
    alpha, gamma, beta, u0, v0, k1, k2 = x0
    A = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    k_distortion = np.array([k1,k2])
    return A, k_distortion

def loss_function(x0, Rt_all, images_corners, world_corners):
    K, k_distortion = convert_xvec_to_Amatrix(x0)
    error_all_images, _ = projection_error(K, k_distortion, Rt_all, images_corners, world_corners)
    return np.array(error_all_images)

def projection_error(K, K_distortion, Rt_all, images_corners, world_corners):
    # print(K_distortion)
    alpha, gamma, beta, u0, v0, k1, k2 = package_x_vector(K, K_distortion)
    num_images = len(images_corners)
    num_corners = len(world_corners)
    error_all_images = np.zeros(num_images)
    reprojected_corners_all = []

    # Convert world_corners to homogeneous coordinates and reshape for broadcasting
    world_corners_homo = np.vstack([world_corners.T, np.zeros(num_corners), np.ones(num_corners)])

    for i in range(num_images):
        Rt = Rt_all[i]
        H = K @ Rt
        img_corners = np.array(images_corners[i])
        img_corners = np.squeeze(img_corners)
        # Project world points to image plane

        proj_coords = H @ world_corners_homo
        u, v = proj_coords[0, :] / proj_coords[2, :], proj_coords[1, :] / proj_coords[2, :]

        # Compute normalized coordinates
        normalized_coords = Rt @ world_corners_homo
        x, y = normalized_coords[0, :] / normalized_coords[2, :], normalized_coords[1, :] / normalized_coords[2, :]
        r = x**2 + y**2

        # Apply distortion
        u_hat = u + (u - u0) * (k1 * r + k2 * r**2)
        v_hat = v + (v - v0) * (k1 * r + k2 * r**2)
        corners_hat = np.vstack([u_hat, v_hat]).T

        reprojected_corners_all.append(corners_hat)
        error = np.linalg.norm(img_corners - corners_hat, axis=1)
        # print(error)
        error_all_images[i] = np.mean(error)
    return error_all_images, np.array(reprojected_corners_all)

def main():
    folder_path = "Calibration_Imgs"
    images = read_caliberation_images(folder_path)
    chessboardSize = (9,6)
    size_of_chessboard_squares_mm = 21.5
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 2), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    worldpoints = objp * size_of_chessboard_squares_mm
    # print(worldpoints)
    imgpoints = getcheckerboradconers(images,chessboardSize)
    homography = calculate_homography(imgpoints,worldpoints)
    B = getmatrix_B(homography)
    K = get_intrinsic_mat(B)
    print("Initial K matrix")
    print(K)
    Rt = get_exterinsic_mat(K,homography)
    # Rt = extrinsics(K,homography)
    intial_k_distortion = np.array([0,0])
    # print("Initial distortion assumed")
    # print(intial_k_distortion)
    x_vector = package_x_vector(K,intial_k_distortion)
    # print(K)
    print("OPTIMIZING....")
    xnew_vector = optimize.least_squares(loss_function, x0=x_vector, method="lm", args=[Rt, imgpoints, worldpoints])
    res = xnew_vector.x
    K_new, K_distortion_new = convert_xvec_to_Amatrix(res)
    print("Optimised K matrix after considering distortion")
    print(K_new)
    print("K distortion paramter")
    print(K_distortion_new)
    avg_error, reprojected_corner = projection_error(K_new, K_distortion_new, Rt, imgpoints, worldpoints)
    print("Reprojection error", np.mean(avg_error))
    dist = np.array([K_distortion_new[0], K_distortion_new[1], 0, 0, 0], dtype = float)
    # dist = np.array([ 2.90493410e-01, -2.42737867e+00 , 0 , 0 ,6.52472281e+00])


    ############## UNDISTORTION #####################################################
    img = cv.imread('/home/ankit/Documents/STUDY/RBE594/HW1/Calibration_Imgs/IMG_20170209_042606.jpg')
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(K_new, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv.undistort(img, K_new, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('caliResult1.png', dst)




if __name__ == "__main__":
    main()