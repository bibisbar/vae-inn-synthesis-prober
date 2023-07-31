import torch


#implement fid score
#
# def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
#                                     cuda=False):
#     model.eval()
#     act=np.empty((len(images), dims))
#
#     if cuda:
#         batch=images.cuda()
#     else:
#         batch=images
#
#     pred=model(batch)[0]
#
#     # If model output is not scalar, apply global spatial average pooling.
#     # This happens if you choose a dimensionality not equal 2048.
#     if pred.size(2) != 1 or pred.size(3) != 1:
#         pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
#
#     act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
#
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma
#
# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
#     Stable version by Dougal J. Sutherland.
#     Params:
#     -- mu1 : Numpy array containing the activations of a layer of the
#              inception net (like returned by the function 'get_predictions')
#              for generated samples.
#     -- mu2   : The sample mean over activations, precalculated on an
#                representative data set.
#     -- sigma1: The covariance matrix over activations for generated samples.
#     -- sigma2: The covariance matrix over activations, precalculated on an
#                representative data set.
#     Returns:
#     --   : The Frechet Distance.
#     """
#
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)
#
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)
#
#     assert mu1.shape == mu2.shape, \
#         'Training and test mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, \
#         'Training and test covariances have different dimensions'
#
#     diff = mu1 - mu2
#
#     # product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = ('fid calculation produces singular product; '
#                'adding %s to diagonal of cov estimates') % eps
#         warnings.warn(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
#
#     # numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError('Imaginary component {}'.format(m))
#         covmean = covmean.real
#
#     tr_covmean = np.trace(covmean)
#
#     return (diff.dot(diff) + np.trace(sigma1) +
#             np.trace(sigma2) - 2 * tr_covmean)
#
# def calculate_fid(images_real, images_fake, model, batch_size=128, dims=2048,
#                   cuda=False):
#     """Calculates the FID of two paths"""
#     mu1, sigma1 = calculate_activation_statistics(images_real, model,
#                                                   batch_size, dims, cuda)
#     mu2, sigma2 = calculate_activation_statistics(images_fake, model,
#                                                   batch_size, dims, cuda)
#     fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
#     return fid_value
#
# def calculate_fid_given_paths(paths, batch_size, cuda, dims):
#     """Calculates the FID of two paths"""
#     for p in paths:
#         if not os.path.exists(p):
#             raise RuntimeError('Invalid path: %s' % p)
#
#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
#
#     model = InceptionV3([block_idx])
#
#     if cuda:
#         model.cuda()
#
#     m1, s1 = calculate_activation_statistics_given_paths(paths[0], model,
#                                                          batch_size,
#                                                          dims, cuda)
#     m2, s2 = calculate_activation_statistics_given_paths(paths[1], model,
#                                                          batch_size,
#                                                          dims, cuda)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)
#
#     return fid_value
#
# def calculate_activation_statistics_given_paths(path, model, batch_size=128,
#                                                 dims=2048, cuda=False):
#     if path.endswith('.npz'):
#         f = np.load(path)
#         m, s = f['mu'][:], f['sigma'][:]
#         f.close()
#     else:
#         path = pathlib.Path(path)
#         files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
#         imgs = np.array([imread(str(fn)).astype(np.float32)
#                          for fn in files])
#         if imgs.shape[3] == 4:
#             imgs = imgs[:, :, :, :3]
#         imgs = imgs.transpose((0, 3, 1, 2))
#         m, s = calculate_activation_statistics(imgs, model, batch_size,
#                                                dims, cuda)
#     return m, s
#
# def get_activations(images, model, batch_size=128, dims=2048, cuda=False,
#                     verbose=False):
#     """Calculates the activations of the pool_3 layer for all images.
#     Params:
#     -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
#                      must lie between 0 and 256.
#     -- model       : Instance of inception model
#     -- batch_size  : the images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size depends
#                      on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- cuda        : If set to True, use GPU
#     -- verbose     : If set to True and parameter out_step is given, the number
#                      of calculated batches is reported.
#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the query
#        tensor.
#     """
#     model.eval()
#
#     d0 = images.shape[0]
#     if batch_size > d0:
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))
#         batch_size = d0
#     n_batches = d0 // batch_size
#     n_used_imgs = n_batches * batch_size
#
#     pred_arr = np.empty((n_used_imgs, dims))
#
#     for i in range(n_batches):
#         if verbose:
#             print('\rPropagating batch %d/%d' % (i + 1, n_batches),
#                   end='', flush=True)
#         start = i * batch_size
#         end = start + batch_size
#
#         batch = images[start:end]
#
#         if cuda:
#             batch = batch.cuda()
#
#         pred = model(batch)[0]
#
#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
#      pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
#
#     if verbose:
#         print(' done')
#
#     return pred_arr
#
