import pandas as pd
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import pylab
import numpy as np

from ml_intro.custutils import to_path

# read data as numpy array with shape n * m * 3, where n an dm is image sizes
image = imread(to_path('parrots.jpg'))
# show image
pylab.imshow(image)
# skimage float format is value in range [0; 1]
img_float = img_as_float(image)
# Reshape array to construct features-objects: every pixel is object, every
# object has 3 features - R, G, B
# reshape with -1: total array size divided by product of all other listed
# dimensions (in this case last dim is 3)
X = img_float.reshape(-1, img_float.shape[-1])
# data frame will be used to groupby by clusters
clust_data = pd.DataFrame(X, columns=['R', 'G', 'B'])
# define psnr - Peak signal-to-noise ratio


def psnr(y_true, y_pred):
    """both params should be vectors, otherwise reshape(-1) performed
    for details see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    in this implementation assumed that image encoded in float format which
    means max signal value is 1.0"""
    if len(y_true.shape) > 1:
        y_true = y_true.reshape(-1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.reshape(-1)
    mse_val = ((y_true - y_pred) ** 2).mean()
    return 10.0 * np.log10(1.0 / mse_val)


# find min cluster number with psnr > 20 db
for num_clust in range(8, 21):
    print(f'Cluster number {num_clust}')
    # Fit k-means
    k_means = KMeans(init='k-means++',
                     random_state=241, n_clusters=num_clust).fit(X)
    clust_data['Cluster label'] = k_means.labels_
    # calculate median and mean
    grps = clust_data.groupby('Cluster label')
    mean_df = grps.mean()
    median_df = grps.median()
    # transform to numpy for performance reasons
    mean_arr = mean_df.values
    median_arr = median_df.values
    # replace objects with same cluster by average and median colour values
    # old way:
    #  mean_clust = clust_data.apply(replace_by(mean_df), axis=1)
    # new way (TO-DO: remove mem overhead):
    mean_clust = np.array([mean_arr[label]
                           for label in k_means.labels_])
    median_clust = np.array([median_arr[label]
                             for label in k_means.labels_])
    mean_psnr = psnr(X, mean_clust)
    median_psnr = psnr(X, median_clust)
    # Plot results
    mean_image = mean_clust.reshape(img_float.shape)
    median_image = median_clust.reshape(img_float.shape)
    pylab.imshow(mean_image)
    pylab.imshow(median_image)
    # pylab.show()
    print(f'Mean image PSNR: {mean_psnr}')
    print(f'Median iamge PSNR: {median_psnr}')
    if max(mean_psnr, median_psnr) > 20.0:
        print(f'PSNR exceeded 20.0 for number of clusters: {num_clust}')
        break

"""
# Plot results
mean_image = mean_clust.reshape(img_float.shape)
median_image = median_clust.reshape(img_float.shape)
pylab.imshow(mean_image)
pylab.imshow(median_image)
# show all plots - not workin in ipython
#pylab.show()"""
