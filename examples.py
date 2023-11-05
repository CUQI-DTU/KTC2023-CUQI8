import numpy as np
import scipy as sp
import matplotlib.image
import os
from skimage.metrics import structural_similarity as ssim

inclusion_list = [1,2,3,4]
categories_list = [1,4,7]

folder_ground_truth = "/app/GroundTruths"
folder_training_data = "/app/TrainingData"
folder_reconstruction = "/app/Output"

folder_output_examples = "/app/results"

def save_as_image(image_array, name):
    matplotlib.image.imsave(name, image_array, vmin = 0, vmax = 2)

def load_mat(folder_name, inclusion, ground_truth=False):
    if ground_truth:
        key = "truth"
        path = folder_name + "/" + "true" + str(inclusion) + ".mat"
    else:
        key = "reconstruction"
        path = folder_name + "/" + str(inclusion) + ".mat"
        
    return np.array(sp.io.loadmat(path).get(key))

def scoring_function(truth, recon):
    truth_c = np.zeros((256,256))
    truth_c[truth == 2] = 1
    recon_c = np.zeros((256,256))
    recon_c[recon == 2] = 1
    s_c = ssim(truth_c, recon_c, data_range = 2.0, gaussian_weights = True,
          K1 = 1*1e-4, K2 = 9*1e-4, sigma = 80.0, win_size = 255)
    
    truth_r = np.zeros((256,256))
    truth_r[truth == 1] = 1
    recon_r = np.zeros((256,256))
    recon_r[recon == 1] = 1
    s_r = ssim(truth_r, recon_r, data_range = 2.0, gaussian_weights = True,
          K1 = 1*1e-4, K2 = 9*1e-4, sigma = 80.0, win_size = 255)
    return 0.5*(s_c+s_r)

if __name__ == "__main__":
    score_matrix = []
    for inclusion in inclusion_list:
        ground_truth = load_mat(folder_ground_truth, inclusion, ground_truth=True)
        file_name = folder_output_examples + "/" + "0" + str(inclusion) + ".png"
        save_as_image(ground_truth, file_name)

    for category in categories_list:
        category_scores = []
        
        print("Reconstructing category number: " + str(category))
        os.system("python3 /app/main.py " + folder_training_data + " " + folder_reconstruction + " " + str(category))
        # main.main(folder_training_data, folder_reconstruction, category)
        for inclusion in inclusion_list:
            ## Save example image
            reconstruction = load_mat(folder_reconstruction, inclusion, ground_truth=False)
            file_name = folder_output_examples + "/" + str(category) + str(inclusion) + ".png"
            save_as_image(reconstruction, file_name)
            
            ## Compute score
            ground_truth = load_mat(folder_ground_truth, inclusion, ground_truth=True)
            category_scores.append(scoring_function(ground_truth, reconstruction))
        
        score_matrix.append(category_scores)

    np.savetxt(folder_output_examples + "/" + "scores.txt", np.array(score_matrix).T, fmt='%4.3f')