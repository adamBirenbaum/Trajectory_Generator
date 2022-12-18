


from make_model_function import make_model
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import re
import seaborn as sns
import shutil
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors as KNN
import sklearn.decomposition

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Reshape, Conv1D, MaxPool1D, Conv1DTranspose
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from tensorflow.keras.metrics import mse

import umap
import umap.plot

physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)




class CustomLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mse_term = mse(y_true, y_pred)

    # get distance 
    pred_dist = tf.norm(y_pred[:,:,:3])

    # take difference of distance to get effective velolocity
    pred_vel = tf.experimental.numpy.diff(pred_dist)

    # actual vel
    true_vel = tf.norm(y_true[:,:,3:6])

    mse_vel = mse()

    rmse = tf.math.sqrt(mse)
    return rmse / tf.reduce_mean(tf.square(y_true)) - 1
 

def make_good_model(nfeat, n_seq):
    # define model
    model = Sequential()
    # encoder
    #model.add(LSTM(100, activation='relu', input_shape=(n_seq,1)))
    model.add(LSTM(100, input_shape=(n_seq,nfeat),return_sequences=True,activation='tanh'))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(50,return_sequences=False,activation='tanh'))
    # decoder
    model.add(RepeatVector(n_seq))
    #model.add(LSTM(100, activation='relu', return_sequences=True))

    model.add(LSTM(50, return_sequences=True,activation='tanh'))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(100,return_sequences=True,activation='tanh'))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(nfeat)))
    model.compile(optimizer='adam', loss='mse',metrics='mse')
    return model




def test_models(model_dir):

    model_files = os.listdir(model_dir)

    for model_file in model_files:
        print('testing {}'.format(model_file))
        fullname = os.path.join(model_dir, model_file)
        model = make_model(fullname, (nsteps, nfeat))

    print('\n\n\nPASS\n\n\n')


def normalize_features(X):


    transformer = StandardScaler().fit(X)
    normalized_features = transformer.transform(X)


    twopi = np.pi * 2
    fourpi = np.pi*4
    
    normalized_features[:,6] = (X[:, 6] + twopi) / fourpi
    normalized_features[:,7] = (X[:, 7] + twopi) / fourpi
    normalized_features[:,8] = (X[:, 8] + twopi) / fourpi


    return normalized_features, transformer
    

def make_autoencoder():

    conv_encoder = Sequential([
        Conv1D(16, kernel_size=3, padding='same', activation='relu', input_shape=[500,9]),
        MaxPool1D(pool_size=2),
        Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        MaxPool1D(pool_size=2)
        ])

    conv_decoder = Sequential([
        Conv1DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=[125, 32]),
        Conv1DTranspose(9, kernel_size=3, strides=2, padding='same', activation='linear')
        ])
    
    conv_ae = Sequential([conv_encoder, conv_decoder])

    conv_ae.compile(loss='mse',optimizer='Adam')

    return conv_encoder, conv_decoder, conv_ae


def generate_traj_umap(X, encoder, decoder, scaler,seed, maxz):
    
    codings = encoder(X).numpy()
    _dims = codings.shape
    
    flattened_codings = codings.reshape((_dims[0], _dims[1] * _dims[2]))
    mapper = umap.UMAP(n_neighbors=200,min_dist=0.0,random_state=seed).fit(flattened_codings)

    # umap.plot.points(mapper)

    # plt.show()

    minx,maxx = np.min(mapper.embedding_[:, 0]), np.max(mapper.embedding_[:,0])
    miny,maxy = np.min(mapper.embedding_[:, 1]), np.max(mapper.embedding_[:,1])

    corners = np.array([
    [minx, miny],  # 1
    [minx, maxy],  # 7
    [maxx,miny],  # 2
    [maxx,maxy],  # 0
    ])

    npoints = 8
    test_pts = np.array([
        (corners[0]*(1-x) + corners[1]*x)*(1-y) +
        (corners[2]*(1-x) + corners[3]*x)*y
        for y in np.linspace(0, 1, npoints)
        for x in np.linspace(0, 1, npoints)
    ])

    inv_transformed_points = mapper.inverse_transform(test_pts)

    

    # Set up the grid
    fig = plt.figure(figsize=(18,12))
    gs = GridSpec(npoints, npoints*2, fig)

    scatter_ax = fig.add_subplot(gs[:, :npoints])
    digit_axes = np.zeros((npoints, npoints), dtype=object)
    for i in range(npoints):
        for j in range(npoints):
            digit_axes[i, j] = fig.add_subplot(gs[i, npoints + j])

    # Use umap.plot to plot to the major axis
    # umap.plot.points(mapper, labels=labels, ax=scatter_ax)
    scatter_ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], s=0.1)
    scatter_ax.set(xticks=[], yticks=[])

    # Plot the locations of the text points
    scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=15)

    # Plot each of the generated digit images
    for i in range(npoints):
        for j in range(npoints):
            inv_points = inv_transformed_points[i*npoints + j].reshape((1,_dims[1],_dims[2]))

            new_traj = decoder(inv_points).numpy()[0]
            new_traj = scaler.inverse_transform(new_traj)
            digit_axes[i, j].plot(new_traj[:,2])
            digit_axes[i, j].set(xticks=[], yticks=[])
            digit_axes[i, j].set_ylim([0,maxz])

    scatter_ax.set_title('UMAP of Latent Space')
    fig.suptitle('Generating Trajectories from uniform samples in UMAP-ed Latent Space')
    fig.savefig(os.path.join('..','plots','umap_plot.png'),dpi=200)
    plt.close('all')

def generate_traj(X,encoder, decoder, scaler, seed, num_gen, k, maxz):

    
    codings = encoder(X).numpy()
    _dims = codings.shape

    flattened_codings = codings.reshape((_dims[0], _dims[1] * _dims[2]))

    pca = sklearn.decomposition.PCA()
    pca.fit(flattened_codings)
    
    n_comp = 2000
    x_reduced = pca.transform(flattened_codings)[:,:n_comp]

    rng = np.random.default_rng(seed)

    ind1 = rng.integers(0,high=x_reduced.shape[0], size = num_gen)
    x1 = x_reduced[ind1]

    neigh = KNN(n_neighbors=k)
    neigh.fit(x_reduced)


    ind2 = rng.integers(0,high=k, size = num_gen)
    
    nearest_neighs = neigh.kneighbors(x1,2,return_distance=False)
    x2_ind = [nearest_neighs[i,_ind2] for i,_ind2 in enumerate(ind2)]

    x2 = x_reduced[x2_ind]

    alpha = rng.random(size=num_gen)

    new_reduced_pts = [_x1*(1-_alpha) + _alpha*_x2 for _alpha, _x1, _x2 in zip(alpha, x1, x2)]

    new_reduced_pts = np.vstack(new_reduced_pts)


    #colors = plt.cm.rainbow(np.linspace(0, 1, num_gen))

    fig, ax = plt.subplots(1,2,figsize=(18,12))
    ax = ax.flatten()
    ax[0].scatter(x_reduced[:,0],x_reduced[:,1],label='original',s=1)
    ax[0].scatter(new_reduced_pts[:,0],new_reduced_pts[:,1],label='generated',s=60)
    ax[0].legend()


    mu = np.mean(flattened_codings, axis=0)
    
    #new_reduced_pts_centered = new_reduced_pts - mu
    new_latent_vecs = np.dot(new_reduced_pts, pca.components_[:n_comp,:])
    new_latent_vecs += mu

    gen_X = decoder(new_latent_vecs.reshape((num_gen,_dims[1],_dims[2]))).numpy()
    
    #gen_X = [np.expand_dims(scaler.inverse_transform(_x),axis=0) for _x in gen_X]
    #gen_X = np.vstack(gen_X)

    ax[1].plot(gen_X[:,:,2].T)
    #ax[1].set_ylim([0, maxz])

    ax[0].set_title('PCA Reduced Latent Space\nw/ Generated Samples')
    ax[1].set_title('Generated Trajectories')
    ax[1].set(xticks=[], yticks=[])
    fig.savefig(os.path.join('..','plots','pca_plot.png'),dpi=200)   

    return gen_X
# (Pdb) xflat = x.reshape((4969,125*32))
# (Pdb) mu = np.mean(xflat)
# (Pdb) pca = sklearn.decompisition.PCA()
# *** AttributeError: module 'sklearn' has no attribute 'decompisition'
# (Pdb) pca = sklearn.decomposition.PCA()
# (Pdb) pca.fit(xflat)
# PCA()
# (Pdb) nComp=2
# (Pdb) xHat = np.dot(pca.transform(xflat)[:,:nComp],pc.components_[:nComp,:])
# *** NameError: name 'pc' is not defined
# (Pdb) xHat = np.dot(pca.transform(xflat)[:,:nComp],pca.components_[:nComp,:])
# (Pdb) mu = np.mean(xflat, axis=0)
# (Pdb) xHat += mu
# (Pdb) xHat.shape
# (4969, 4000)


if __name__ == '__main__':



    iter_name = 'Test'
    max_len = 500

    
    


    train_size = 0.9
    epochs=500
    batch_size = 256
    nfeat=9


    feat_path = os.path.join('..','outputs','Processed', iter_name, 'processed_features.txt')



    
    out_train_dir = os.path.join('..','outputs','training', iter_name)


    tensorboard_dir = os.path.join(out_train_dir, 'tensorboard')
    plot_dir = os.path.join(out_train_dir, 'plots')

    
    tb_dir = os.path.join(tensorboard_dir, 'testing')
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)


    os.makedirs(out_train_dir,exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)



    X = np.loadtxt(feat_path)


    quant = .95
    minz, maxz = np.quantile(X[:,2],1-quant), np.quantile(X[:,2],quant)
    
    X, scaler = normalize_features(X)
    

    
    nruns = int(X.shape[0] / max_len)
    nsteps = max_len
    
    X = X.reshape((nruns, nsteps, nfeat))
    
    is_nan = np.any(np.isnan(X))
    
    if is_nan:
        delete_ind = np.where(np.isnan(X))[0][0]
        X = np.delete(X,delete_ind,axis=0)

    seed = 100

    X_train, X_test, y_train, y_test = train_test_split(X, X, train_size=train_size,random_state=seed)
    

    conv_encoder, conv_decoder, model = make_autoencoder()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir)
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,  validation_data=(X_test, X_test), callbacks=[tensorboard_callback])
        

    
    num_gen = 10
    k = 2

    #new_traj = generate_traj_umap(X_train,conv_encoder, conv_decoder, scaler,seed, maxz)

    new_traj = generate_traj(X,conv_encoder, conv_decoder, scaler,seed, num_gen, k,maxz)

    breakpoint()
