import numpy
import numpy.matlib
import time
from Log import log
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import misc
from sklearn.cluster import KMeans, MeanShift
import os
import cv2
import matplotlib.image as mpimg


def view_latent_space(engine,network,data,epoch):
    m =data.num_train_id
    counts = data.train_counts.eval()
    n = counts.sum()
    idx_placeholder = data.idx_placeholder
    batch_size = network.batch_size
    debug = network.tags
    path = data.test_case
    # path_name = "class_data"
    path_name = engine.config.unicode("train_folder", "outputTriplet")

    out_layer_name = engine.config.unicode("output_embedding_layer", "outputTriplet")
    out_layer = network.tower_layers[0][out_layer_name]
    assert len(out_layer.outputs) == 1
    out_feature = out_layer.outputs[0]
    out_feature_size = out_layer.n_features

    all_val = numpy.empty([0, out_feature_size])
    col_vec = numpy.empty([0, 1])
    sing_val = numpy.empty([0, out_feature_size])
    sing_col = numpy.empty([0, 1])
    vars = numpy.empty([0,1])

    get_images = data.images
    all_images = numpy.empty([0,64,64,3])

    for i in range(len(counts)):
        curr_val = numpy.empty([0, out_feature_size])
        curr_col_vec = numpy.empty([0, 1])
        idx = 0
        while idx < counts[i]:
            start = time.time()
            idx_value = [idx, min(idx + batch_size, counts[i]), i, 0]

            feature_val, msg, images = engine.session.run([out_feature, debug, get_images],
                                                feed_dict={idx_placeholder: idx_value, path: path_name})
            all_images = numpy.concatenate((all_images,images),axis=0)
            curr_val = numpy.concatenate((curr_val, feature_val), axis=0)

            end = time.time()
            elapsed = end - start
            print(min(idx + batch_size, counts[i]), '/', counts[i],",", i+1,"/", len(counts), "elapsed", elapsed, file=log.v5)
            idx += batch_size
            curr_col_vec = i*numpy.ones([counts[i],1])



        all_val = numpy.concatenate((all_val,curr_val))
        col_vec = numpy.concatenate((col_vec, curr_col_vec))
        sing_val = numpy.concatenate((sing_val, numpy.expand_dims(numpy.mean(curr_val,0),0)))
        sing_col = numpy.concatenate((sing_col, numpy.expand_dims(numpy.expand_dims(numpy.array(i),0),0)))

        curr_var = numpy.var(curr_val,0).mean()
        vars = numpy.concatenate((vars, numpy.expand_dims(numpy.expand_dims(numpy.array(curr_var),0),0)))

    int_var = numpy.mean(vars)
    ext_var = numpy.var(sing_val,0).mean()

    rank_out = "%.4f " % int_var + "%.4f " % ext_var

    measures = {}
    measures["ranks"] = rank_out

    pca = PCA(n_components=2)
    new_all_val = pca.fit_transform(all_val)

    pca = PCA(n_components=32)
    pca.fit(all_val)
    # PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
    #     svd_solver='auto', tol=0.0, whiten=False)
    print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    add_dim_probs = numpy.array(pca.explained_variance_ratio_)
    dim_probs = add_dim_probs.cumsum()
    print(dim_probs)
    # dims = numpy.nonzero(dim_probs>0.9)
    dims = numpy.nonzero(add_dim_probs < 0.03)

    print("number of dims = " + str(dims[0][0]))
    pca = PCA(n_components=dims[0][0])
    cluster_val = pca.fit_transform(all_val)

    # num = 50
    # kmeans = KMeans(n_clusters=num, random_state=0).fit_predict(cluster_val)
    # print(kmeans)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
    cluster_labels = clusterer.fit_predict(cluster_val)
    print(cluster_labels)
    print("number of clusters = " + str(max(cluster_labels)+1))
    # kmeans = cluster_labels
    col_vec = cluster_labels


    # new_all_val = TSNE(n_components=2).fit_transform(all_val)

    # new_all_val = numpy.tanh(all_val)
    # new_sing_val = numpy.tanh(sing_val)

    fig1 = plt.figure(1)
    # plt.scatter(all_val[:, 0], all_val[:, 1], c=numpy.array(col_vec), cmap=plt.get_cmap("rainbow"),marker='.')
    plt.scatter(new_all_val[:, 1], -1*new_all_val[:, 0], c=numpy.array(col_vec), cmap=plt.get_cmap("rainbow"), marker='.')
    fig1.savefig('/home/dlrc/dlrc17-gdk/ext/OnAVOS/class_output/' + str(epoch) + '_all.png')
    plt.close(fig1)
    # plt.show()

    # new_sing_val = pca.transform(sing_val)
    # fig2 = plt.figure(2)
    # plt.scatter(new_sing_val[:, 0], new_sing_val[:, 1], c=numpy.array(sing_col), cmap=plt.get_cmap("rainbow"),marker='.')
    # fig2.savefig('/home/dlrc/dlrc17-gdk/ext/OnAVOS/class_output/' + str(epoch) + '_sing.png')
    # plt.close(fig2)
    # # plt.show()

    fig3 = plt.figure(3)
    tot_im_size = 1024
    tot_im = numpy.zeros((tot_im_size,tot_im_size,3))
    fig3, axes = plt.subplots(1, 1, figsize=(20, 20))
    ss = 64
    for i in range(new_all_val.shape[0]):
        ix = ((new_all_val[i] - numpy.min(new_all_val)) / (numpy.max(new_all_val) - numpy.min(new_all_val)) * (tot_im_size-ss)).astype(int)
        tot_im[ix[0]:ix[0] + ss, ix[1]:ix[1] + ss] = misc.imresize(all_images[i,:,:,:].reshape((64, 64, 3)), (ss, ss))
    axes.imshow(tot_im * 255)
    fig3.savefig('/home/dlrc/dlrc17-gdk/ext/OnAVOS/class_output/' + str(epoch) + '_bricks.png')
    plt.close(fig3)


    num = 6
    kmeans = KMeans(n_clusters=num, random_state=0).fit_predict(all_val)
    print("kmeans: " + str(kmeans))

    # bandwidth = 5
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # cluster = ms.fit_predict(all_val)
    # print(cluster)
    # kmeans = cluster[cluster!=0]

    im_fol = engine.config.unicode("data_dir", "outputTriplet/") + engine.config.unicode("train_folder", "outputTriplet") + '/'
    print(im_fol)
    # im_fol = "/home/dlrc/Documents/Segment/OnAVOSold/custom_dataset/JPEGImages/480p/"
    ims = []
    for im_filename in sorted(os.listdir(im_fol)):
        cur_im = mpimg.imread(im_fol+im_filename)
        # cur_im = cv2.imread(im_fol+im_filename)
        # cur_im = misc.imread(im_fol + im_filename)
        ims.append(cur_im)



    # fig4 = plt.figure(4)
    for label in set(kmeans):
        # print('label',label)
        # tmp_images = list(numpy.array(all_images)[kmeans == label])
        tmp_images = list(numpy.array(ims)[kmeans == label])
        n_rows = int(numpy.ceil(numpy.sqrt(len(tmp_images))))
        #plt.figure(figsize=(15, 15))
        fig4, axs = plt.subplots(n_rows,n_rows,figsize=(20, 20))
        axlist = []
        for i, img in enumerate(tmp_images):
            # print(i)
            _x,_y = numpy.unravel_index(i, (n_rows,n_rows))
            ax = axs[_x,_y]
            ax.set_xticks([])
            ax.set_yticks([])
            axlist.append(ax)
            _img = tmp_images[i].reshape((64, 64, 3))
            # print(_img)
            ax.imshow(_img)
        for i in range(n_rows):
            for j in range(n_rows):
                if axlist.count(axs[i,j])==0:
                    axs[i, j].set_axis_off()
        fig4.savefig('/home/dlrc/dlrc17-gdk/ext/OnAVOS/class_output/' + str(epoch) + '_' + str(label) + '_classes.png')
        plt.close(fig4)

    # fig4 = plt.figure(4)
    for label in set(cluster_labels):
        # print('label',label)
        # tmp_images = list(numpy.array(all_images)[kmeans == label])
        tmp_images = list(numpy.array(ims)[cluster_labels == label])
        n_rows = int(numpy.ceil(numpy.sqrt(len(tmp_images))))
        #plt.figure(figsize=(15, 15))
        fig5, axs = plt.subplots(n_rows,n_rows,figsize=(20, 20))
        axlist = []
        for i, img in enumerate(tmp_images):
            # print(i)
            _x,_y = numpy.unravel_index(i, (n_rows,n_rows))
            ax = axs[_x,_y]
            ax.set_xticks([])
            ax.set_yticks([])
            axlist.append(ax)
            _img = tmp_images[i].reshape((64, 64, 3))
            # print(_img)
            ax.imshow(_img)
        for i in range(n_rows):
            for j in range(n_rows):
                if axlist.count(axs[i,j])==0:
                    axs[i, j].set_axis_off()
        fig5.savefig('/home/dlrc/dlrc17-gdk/ext/OnAVOS/class_output/' + str(epoch) + '_' + 'kmeans' + str(label) + '_classes.png')
        plt.close(fig5)

    return 0,measures





def do_cmc_validation(engine,network,data):
  m = data.num_test_id
  n = m * m
  idx_placeholder = data.idx_placeholder
  batch_size = network.batch_size
  debug = network.tags
  path = data.test_case
  end_net = data.use_end_network
  rank_out = ""
  errors = ""
  measures = {}
  merge_type = engine.config.unicode("merge_type", "")

  out_layer_name = engine.config.unicode("output_embedding_layer","fc1")
  out_layer = network.tower_layers[0][out_layer_name]
  assert len(out_layer.outputs) == 1
  out_feature = out_layer.outputs[0]
  out_feature_size = out_layer.n_features

  test_cases = engine.config.unicode_list("test_cases", [])

  for test_case in test_cases:
    errs = 0
    y_vals = numpy.empty([0,1])
    probe = numpy.empty([0, out_feature_size])
    gallery = numpy.empty([0, out_feature_size])

    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + batch_size, m),1,0]

      feature_val, msg = engine.session.run([out_feature, debug],
                                            feed_dict={idx_placeholder: idx_value, path: test_case, end_net: False})
      probe = numpy.concatenate((probe, feature_val), axis=0)

      end = time.time()
      elapsed = end - start
      print (min(idx + batch_size, m), '/', m, "elapsed", elapsed)
      idx += batch_size

    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + batch_size, m), 1, 1]

      feature_val, msg = engine.session.run([out_feature, debug],
                                            feed_dict={idx_placeholder: idx_value, path: test_case, end_net: False})
      gallery = numpy.concatenate((gallery, feature_val), axis=0)

      end = time.time()
      elapsed = end - start
      print (min(idx + batch_size, m), '/', m, "elapsed", elapsed)
      idx += batch_size

    start = time.time()
    for pdx in range(m):
      idx = 0
      while idx < m:
        idx_value = [idx, min(idx + batch_size, m), pdx, 1]
        r = numpy.arange(idx_value[0], idx_value[1])
        q = (pdx,) * (min(idx + batch_size, m) - idx)

        if data.validation_mode == "similarity":

          y = network.y_softmax
          e = network.measures_accumulated
          in_layer_name = engine.config.unicode("input_embedding_layer", "siam_concat")
          in_layer = network.tower_layers[0][in_layer_name]
          assert len(in_layer.outputs) == 1
          in_feature = in_layer.outputs[0]

          if merge_type == "add":
            feature_val = probe[q, :] + gallery[r, :]
          elif merge_type == "subtract":
            feature_val = probe[q, :] - gallery[r, :]
          elif merge_type == "abs_subtract":
            feature_val = numpy.abs(probe[q, :] - gallery[r, :])
          else: # merge_type == "concat":
            feature_val = numpy.concatenate((probe[q, :], gallery[r, :]), axis=1)

          y_val, err = engine.session.run([y, e], feed_dict={idx_placeholder: idx_value, in_feature: feature_val, end_net: True,path: test_case})
          y_val = y_val[:,0:1]
          errs += err["errors"]

        else: # data.validation_mode == "embedding":
          y_val = numpy.linalg.norm(probe[q,:] - gallery[r,:],axis=1)
          y_val = numpy.reshape(y_val,[y_val.size,1])

        y_vals = numpy.concatenate((y_vals, y_val), axis=0)
        idx += batch_size

    y_vals1 = y_vals
    Apsum = 0
    ranks = numpy.zeros(m)
    for i in range(m):
      r = numpy.arange(m * i, m * (i + 1))
      I = numpy.identity(m)
      corr = I[:, i]
      tab = numpy.column_stack((y_vals1[r], corr))
      id = numpy.argsort(y_vals1[r], axis=0)
      tab = tab[id, :]
      pos = numpy.where(tab[:,0, 1])[0]
      ranks[i] = pos[0] + 1
      Ap = numpy.zeros(1)
      f = numpy.zeros(1)
      for j in range(pos.size):
        f += 1
        Ap += f / (pos[j] + 1)

      Apsum += Ap

    mAp = Apsum / m
    cmc = numpy.zeros(m)
    for i in range(m):
      cmc[i] = 100 / m * ranks[ranks <= i + 1].size

    rank1 = cmc[0]
    rank5 = cmc[4]
    error = errs / n

    errors += "%.3f " % mAp
    rank_out += "%.1f " % rank1 + "%.1f " % rank5

    measures = {}
    measures["ranks"] = rank_out

    end = time.time()
    elapsed = end - start
    print (test_case, "elapsed", elapsed)

  return errors, measures