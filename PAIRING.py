import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from time import time
import gc
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import umap
import umap.plot

import progeny
import scanpy as sc
import scanpy.external as sce
sc.settings.verbosity = 3 
import anndata as ad

import SEACells
import qnorm

from model_z1536_klnormcrvae_shxpr_vecont02mean_trideltemper5_nonoise_supcon10_delcell_delcontnoloss.pgan_vae_mapping_pr_disc2_xpr_basetf2 import *

def mean_pairwise_diff(term1, term2):
    len1 = term1.shape[0]
    len2 = term2.shape[0]
    
    diff = []
    for i in range(len1):
        for j in range(len2):
            diff.append(term1[i] - term2[j])
            
    return np.mean(diff, axis=0, keepdims=True)
    # return diff


def mean_pairwise_diff_tf(term1, term2):
    diff = term1[None,:,:] - term2[:,None,:]
    diff = tf.reshape(diff, [-1, tf.shape(term1)[-1]])
            
    return tf.reduce_mean(diff, axis=0, keepdims=True)

class PAIRING:
    def __init__(self, model_path='model_z1536_klnormcrvae_shxpr_vecont02mean_trideltemper5_nonoise_supcon10_delcell_delcontnoloss/model-1750.ckpt'):
        print('Data load...')
        train_data = pd.read_csv('l1000_data/p1shp2xpr/training_data_p1.csv', index_col=None)
        geneid = np.array(train_data.columns[2:]).astype('int')
        train_data_sh = pd.read_csv('l1000_data/p1shp2xpr/training_data_sh_p1.csv', index_col=None)
        train_data2 = pd.read_csv('l1000_data/p1shp2xpr/training_data_p2.csv', index_col=None)
        # train_data_xpr = pd.read_csv('l1000_data/p1shp2xpr/training_data_xpr_p2.csv', index_col=None)
        train_data_ve = pd.read_csv('l1000_data/p1shp2xpr/training_data_vehicle.csv', index_col=None)
        train_data_ve = train_data_ve.loc[train_data_ve['pert_iname']=='DMSO']
        train_data_ve.loc[:,'pert_iname'] = 'control'
        
        self.train_data = pd.concat((train_data, train_data_sh, train_data2, train_data_ve))
        # train_data = pd.concat((train_data, train_data2))
        #train_data = pd.concat((basalccl, train_data_sh))
        
        # val_data = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p1.csv', index_col=None)
        # val_data2 = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p2.csv', index_col=None)
        # val_data_ve = pd.read_csv('l1000_data/p1shp2xpr/validation_data_vehicle.csv', index_col=None)
        # val_data_ve = val_data_ve.loc[val_data_ve['pert_iname']=='DMSO']
        # val_data_ve.loc[:,'pert_iname'] = 'control'
        # self.val_data = pd.concat((val_data, val_data2, val_data_ve))
        self.infoidx = infoidx = 2
        
        self.train_data_info = self.train_data.iloc[:,:infoidx].values
        self.train_data = self.train_data.iloc[:,infoidx:].values
        # self.val_data_info = self.val_data.iloc[:,:infoidx].values
        # self.val_data = self.val_data.iloc[:,infoidx:].values
        
        self.train_data_sh = pd.concat((train_data_sh, train_data_ve))
        self.train_data_sh_info = self.train_data_sh.iloc[:,:infoidx].values
        self.train_data_sh = self.train_data_sh.iloc[:,infoidx:].values
        
        self.train_data_comp = pd.concat((train_data, train_data2, train_data_ve))
        self.train_data_comp_info = self.train_data_comp.iloc[:,:infoidx].values
        self.train_data_comp = self.train_data_comp.iloc[:,infoidx:].values
        
        geneinfo = pd.read_csv('l1000_data/GSE92742_Broad_LINCS_gene_info.txt', sep='\t')
        genemapper = pd.Series(data=geneinfo['pr_gene_symbol'].values, index=geneinfo['pr_gene_id'].values)
        self.genesym = pd.Series(geneid).map(genemapper).values
        self.pmodel = progeny.load_model(organism='Human', top=2300)
        
        mol_meta = pd.read_csv('l1000_data/LINCS_small_molecules.tsv', sep='\t')
        mol_meta.index = mol_meta['pert_name']
        _, uid = np.unique(mol_meta.index, return_index=True)
        self.mol_meta = mol_meta.iloc[uid]
        self.mol_meta_tar = self.mol_meta.loc[self.mol_meta['target']!='-',:]
        
        
        train_data_ve_info_ = train_data_ve.iloc[:,:infoidx].values
        train_data_ve_ = train_data_ve.iloc[:,infoidx:].values
        vecells = np.unique(train_data_ve_info_[:,0])
        n_clus = 3
        train_data_ve_clus, train_data_ve_clus_info = [], []
        for vc in vecells:
            vcexp = train_data_ve_[train_data_ve_info_[:,0]==vc]
            if vcexp.shape[0] > 10:
                #km = KMeans(n_clusters=n_clus, random_state=0, n_init="auto").fit_predict(vcexp)
                #train_data_ve_clus_info.append([vc]*n_clus)
                km = HDBSCAN(min_cluster_size=5).fit_predict(vcexp)
                #km = HDBSCAN(min_cluster_size=5).fit_predict(umap.UMAP().fit(vcexp).embedding_)
                if np.sum(km==-1) > vcexp.shape[0]*0.1:
                    km[:] = 0
                    n_clus=1
                else:
                    n_clus = np.max(km[km>=0])+1
                train_data_ve_clus.append(np.concatenate([vcexp[km==_].mean(axis=0, keepdims=True) for _ in range(n_clus)], axis=0))
                #train_data_ve_clus.append(vcexp[km==np.argmax([np.sum(km==_) for _ in range(n_clus)])].mean(axis=0, keepdims=True))
                train_data_ve_clus_info.append([vc]*n_clus)
            else:
                train_data_ve_clus.append(vcexp)
                train_data_ve_clus_info.append([vc]*vcexp.shape[0])
        self.train_data_ve_ = np.concatenate(train_data_ve_clus, axis=0)
        self.train_data_ve_info_ = np.concatenate(train_data_ve_clus_info, axis=0).reshape([-1,1])


        self.landmark_genes = pd.read_csv('l1000_data/genelist.csv')
        self.adata_train = ad.AnnData(pd.DataFrame(self.train_data, index=np.arange(self.train_data.shape[0]).astype('str'), columns=self.landmark_genes['pr_gene_symbol']))
        self.adata_train.obs = pd.DataFrame(self.train_data_info, index=np.arange(self.train_data.shape[0]).astype('str'), columns=['sample', 'perturbation'])
        self.target_dist = np.median(self.adata_train.X, axis=0)
        
        self.ols_weights = pd.read_csv('l1000_data/ols_weights_mat.csv', index_col=0)
        self.Ws = self.ols_weights.loc[:,self.landmark_genes['pr_gene_symbol']]
        self.b = self.ols_weights.loc[:,'OFFSET']
        
        print('Done')
        
        # model params
        mp = {
            'z_dim':1536,
            'learning_rate':5e-4,
            'dropout_rate':0.2,
            'lamb_gp':10,
            'lamb_recon':1,
            'lamb_triple':1,
            'lamb_delta':1,
            'Diters':5,
            'enc_unit1':1024,
            'enc_unit2':512,
            'enc_unit3':256,
            'gen_unit1':512,
            'gen_unit2':512,
            'gen_unit3':1024,
            'disc_unit1':1024,
            'disc_unit2':512,
            'disc_unit3':256,
            'train_cell_num':18,
            'train_pert_num':8532
            }
        
        print('Model load...')
        self.model = PGAN(x_dim=self.train_data.shape[1], xa_dim=self.pmodel.shape[1], **mp)
        
        with tf.device('/cpu:0'):
            self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
            
            seconfig = tf.compat.v1.ConfigProto(allow_soft_placement = True)
            seconfig.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=seconfig)
            
            self.saver.restore(self.sess, model_path)
            
        with tf.device('/cpu:0'):
            self.model.t_term5 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.model.x_dim], name="t_term5") #pert
            self.model.t_term6 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.model.x_dim], name="t_term6") #control
            self.model.pertalpha = tf.compat.v1.placeholder(tf.float32, shape = [None, self.model.z_dim], name="pertalpha")
        
        #VAE many sample - pairwise difference##########################
        with tf.device('/cpu:0'):
            num_sample = 10
            t1_mean, t1_std, t1_encode = self.model.encoder(self.model.t_term1)
            t1_encode = t1_mean[tf.newaxis,:,:] + t1_std[tf.newaxis,:,:] * tf.random.normal(shape=[num_sample, tf.shape(t1_mean)[0], self.model.z_dim])
            t5_mean, t5_std, t5_encode = self.model.encoder(self.model.t_term5)
            t5_encode = t5_mean[tf.newaxis,:,:] + t5_std[tf.newaxis,:,:] * tf.random.normal(shape=[num_sample, tf.shape(t5_mean)[0], self.model.z_dim])
            t6_mean, t6_std, t6_encode = self.model.encoder(self.model.t_term6)
            t6_encode = t6_mean[tf.newaxis,:,:] + t6_std[tf.newaxis,:,:] * tf.random.normal(shape=[num_sample, tf.shape(t6_mean)[0], self.model.z_dim])
            self.model.t1_mapping = self.model.mapper(tf.reshape(t1_encode, [-1,self.model.z_dim]))
            self.model.t5_mapping = self.model.mapper(tf.reshape(t5_encode, [-1,self.model.z_dim]))
            self.model.t6_mapping = self.model.mapper(tf.reshape(t6_encode, [-1,self.model.z_dim]))
            
            self.model.perteffect = mean_pairwise_diff_tf(self.model.t5_mapping, self.model.t6_mapping)
            self.model.arithmetic_triple_mean = self.model.t1_mapping + self.model.pertalpha
            self.model.gen_arith_mean = self.model.generatorDropOut(self.model.arithmetic_triple_mean)
        ##############################################
        
        with tf.device('/cpu:0'):
            self.delta = tf.compat.v1.placeholder(tf.float32, shape = [], name="delta")
            latent_interpolation = self.model.t5_mapping + self.model.pertalpha * self.delta
            self.model.generated_interpolation = self.model.generatorDropOut(latent_interpolation)

        print('Done')
    
    def sc_preprocess_and_metacell(self, new_data):
        """
        Preprocess single cell data.
        normalize_total - log1p - SEACells

        Parameters
        ----------
        new_data : AnnData
            shape n_samples x n_genes containing single-cell gene expression values.

        Returns
        -------
        adata_scrna_landmark : np.array
            shape n_samples x n_genes containing metacell gene expression values.

        """
        
        raw_ad = sc.AnnData(new_data.X)
        raw_ad.obs_names, raw_ad.var_names = new_data.obs_names, new_data.var_names
        new_data.raw = raw_ad

        # Normalize cells, log transform and compute highly variable genes
        sc.pp.normalize_total(new_data)
        sc.pp.log1p(new_data, base=2)
        
        allcase = np.unique(new_data.obs['Sample']) # Sample = Patient identifier + state
        
        summ_house = []
        for aci, ac in enumerate(allcase):
            print(aci,'/',len(allcase))
            
            cur_ad = new_data[new_data.obs['Sample']==ac]
            sc.tl.pca(cur_ad)
        
            if cur_ad.shape[0] < 100:
                n_SEACells = int(np.ceil(cur_ad.shape[0] / 10))
            else:
                n_SEACells = int(np.ceil(cur_ad.shape[0] / 75))
            build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
                                      # This would be replaced by 'X_svd' for ATAC data
        
            ## Additional parameters
            n_waypoint_eigs = np.min([n_SEACells, 10]) # Number of eigenvalues to consider when initializing metacells
        
            scmodel = SEACells.core.SEACells(cur_ad, 
                      build_kernel_on=build_kernel_on, 
                      n_SEACells=n_SEACells, 
                      n_waypoint_eigs=n_waypoint_eigs,
                      convergence_epsilon = 1e-5,
                      use_gpu=False)
        
            scmodel.construct_kernel_matrix()
            
            scmodel.initialize_archetypes()

            scmodel.fit(min_iter=10, max_iter=100000)
            
            SEACell_ad = SEACells.core.summarize_by_SEACell(cur_ad, SEACells_label='SEACell', summarize_layer='raw')
            
            SEACell_ad.obs['Sample'] = cur_ad.obs.groupby('SEACell').apply(lambda x: x['Sample'].mode().iloc[0]).loc[SEACell_ad.obs_names]
            # SEACell_ad.obs['Patient'] = cur_ad.obs.groupby('SEACell').apply(lambda x: x['Patient'].mode().iloc[0]).loc[SEACell_ad.obs_names]
            SEACell_ad.obs['state'] = cur_ad.obs.groupby('SEACell').apply(lambda x: x['state'].mode().iloc[0]).loc[SEACell_ad.obs_names]
            
            summ_house.append(SEACell_ad)
            
        SEACell_ad = ad.concat(summ_house)
        SEACell_ad.obs_names_make_unique()
        SEACell_ad_pu = SEACell_ad
        SEACell_ad_pu.raw = SEACell_ad_pu.copy()
        sc.pp.normalize_total(SEACell_ad_pu)
        sc.pp.log1p(SEACell_ad_pu, base=2)
        SEACell_ad_pu = SEACell_ad_pu[:,np.any(SEACell_ad_pu.X.toarray()!=0, axis=0)]
        
        sce.pp.magic(SEACell_ad_pu, name_list='all_genes')
        
        interidx = np.in1d(SEACell_ad_pu.var.index, self.landmark_genes['pr_gene_symbol'])
        adata_landmark = SEACell_ad_pu[:,interidx]
        
        scrna_landmark = pd.DataFrame(np.zeros([adata_landmark.shape[0], self.landmark_genes.shape[0]]), index=adata_landmark.obs.index, columns=self.landmark_genes['pr_gene_symbol'])
        seqval = adata_landmark.X.toarray()
        scrna_landmark.loc[:,adata_landmark.var.index] = seqval
        
        adata_scrna_landmark = ad.AnnData(scrna_landmark)
        adata_scrna_landmark.obs = adata_landmark.obs.astype('category')
    
        return adata_scrna_landmark
        
        
    def preprocess(self, new_data, do_magic=False, umap_plot=False, is_single_cell=False):
        """
        Preprocess data.
        combat - quantile normalization

        Parameters
        ----------
        new_data : AnnData
            shape n_samples x n_genes containing gene expression values.
        do_magic : bool, optional
            Imputation by magic. The default is False.
        umap_plot : bool, optional
            Plot preprocessed gene expression values using UMAP. The default is False.
        is_single_cell : bool, optional
            True if the new_data is scRNA-seq. The default is False.

        Returns
        -------
        corrected : np.array
            shape n_samples x n_genes containing gene expression values.
        new_data.obs : pd.DataFrame
            Metadata.

        """
        
        if is_single_cell:
            # single-cell data have to have three columns in obs [Sample, state], where [Sample = Patient identifier + state]
            # obs.index have to be gene symbol
            do_magic=False
            print('Metacell...')
            new_data = self.sc_preprocess_and_metacell(new_data)
            print('Metacell... Done')
            
        if do_magic:
            sce.pp.magic(new_data, name_list='all_genes')
        
        adata_mc_concat = self.adata_train.concatenate(new_data, batch_categories=['ref', 'new'])
        adata_mc_concat_combat = sc.pp.combat(adata_mc_concat, key='batch', inplace=False)
        corrected = qnorm.quantile_normalize(adata_mc_concat_combat[-new_data.shape[0]:,:].T, target=self.target_dist).T
        
        if umap_plot:
            label = new_data.obs['state']
            mapper = umap.UMAP(random_state=0).fit(corrected)
            df = pd.DataFrame({'x':mapper.embedding_[:,0], 'y':mapper.embedding_[:,1], 'cls':label})
            plt.figure()
            sns.scatterplot(x='x', y='y', hue='cls', alpha=0.5,data=df)

        gc.collect()
        return corrected, new_data.obs

    def find_perturbation(self, initial_state, desired_state, pert_type='all', save_path=None):
        """
        Find perturbations capable of inducing desired state from initial state.

        Parameters
        ----------
        initial_state : np.array
            shape n_samples x n_genes containing gene expression values.
        desired_state : np.array
            shape n_samples x n_genes containing gene expression values.
        pert_type : str, optional
            Type of perturbations. Find promising perturbations on {'all', 'compound', 'shRNA'}. The default is 'all'.
        save_path : str, optional
            Path to save directory and filename. Result will not be saved if save_path is None.. The default is None.

        Raises
        ------
        ValueError
            Raise error when the "pert_type" is not [all, compound, shRNA].

        Returns
        -------
        result : pd.DataFrame
            Perturbation agents and their scores.

        """
        if pert_type not in ['all', 'compound', 'shRNA']:
            raise ValueError('pert_type: [all, compound, shRNA]')

        if pert_type == 'all':
            train_data_use = self.train_data
            train_data_info_use = self.train_data_info
        elif pert_type == 'compound':
            train_data_use = self.train_data_comp
            train_data_info_use = self.train_data_comp_info
        elif pert_type == 'shRNA':
            train_data_use = self.train_data_sh
            train_data_info_use = self.train_data_sh_info
        
        sc.settings.verbosity = 0

        totalccl = np.unique(train_data_info_use[:,0])
        totalpert = np.unique(train_data_info_use[:,1])
        totalpert = totalpert[totalpert!='control']

        Aalpha = initial_state
        Abeta = desired_state
        
        rankperf_house = []
 
        np.random.seed(0)
        
        print('Initial state shape: ', Aalpha.shape, 'Desired state shape: ', Abeta.shape)
    
        alpha_house = []
        liveidx = np.full(len(totalpert), True)
        for _ in tqdm(range(len(totalpert))):
            p = totalpert[_]
            foralpha, foralphacontrol, aapcc = [], [], []
            for ccl in totalccl:
                if np.where((train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]==p))[0].size > 0:
                    if not np.any((train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]=='control')):
                        continue
                    np.random.seed(0)
                    #foralpha.append(train_data_use[(train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]==p),:])
                    fa = train_data_use[(train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]==p),:]
                    if (fa.shape[0] > 1) & (fa.shape[0] < 100):
                        fa_aug = np.repeat(fa, 2, axis=0) + np.random.normal(0, fa.std(axis=0, keepdims=True), size=(fa.shape[0]*2, fa.shape[1]))
                        fa = np.concatenate((fa, fa_aug), axis=0)
                    foralpha.append(fa)
                    fac = self.train_data_ve_[self.train_data_ve_info_[:,0]==ccl,:]
                    if fac.shape[0] > 1:
                        fac_aug = np.repeat(fac, 2, axis=0) + np.random.normal(0, fac.std(axis=0, keepdims=True), size=(fac.shape[0]*2, fac.shape[1]))
                        fac = np.concatenate((fac, fac_aug), axis=0)  
                    foralphacontrol.append(fac)
                    #foralphacontrol.append(basal_exp[basal_info[:,0]==ccl])
                    
            if len(foralpha) > 0:
                alpha = []
                #amapping = self.sess.run(self.model.t1_mapping, feed_dict={self.model.t_term1:Aalpha, self.model.is_training:False})
                for cc in range(len(foralpha)):
                    feed_dict = {self.model.t_term5:foralpha[cc], self.model.t_term6:foralphacontrol[cc], self.model.is_training:False}
                    cmapping, calpha = self.sess.run([self.model.t6_mapping, self.model.perteffect], feed_dict=feed_dict)
                    # pmapping, cmapping = self.sess.run([self.model.t5_mapping, self.model.t6_mapping], feed_dict=feed_dict)
                    # calpha = mean_pairwise_diff(pmapping, cmapping)
                    alpha.append(calpha)
    
                    aapcc.append(-np.mean(cdist(Aalpha, foralphacontrol[cc], metric='correlation').ravel()))
                    #aapcc.append(-np.mean(cdist(amapping, cmapping, metric='euclidean').ravel()))
                    #aapcc.append(-np.mean(cdist(amapping, cmapping, metric='minkowski', p=1).ravel()))
                    #aapcc.append(1-np.mean(cdist(amapping, cmapping, metric='cosine').ravel()))   #cosine, correlation
    
    
                e_x = np.exp(aapcc - np.max(aapcc))# ** 5
                pccweight = e_x / e_x.sum()
                alpha = np.average(alpha, axis=0, weights=pccweight)
                alpha_house.append(alpha)
                #alpha = np.mean(alpha, axis=0)
                #alpha = alpha[np.argmax(aapcc)]

        
        latent_Abeta, latent_Aalpha = self.sess.run([self.model.t5_mapping, self.model.t6_mapping], feed_dict={self.model.t_term5:Abeta, self.model.t_term6:Aalpha, self.model.is_training:False})
        target_alpha = mean_pairwise_diff(latent_Abeta, latent_Aalpha)
        ah = np.concatenate(alpha_house, axis=0)[liveidx]
    
        cors_a = 1 - cdist(target_alpha, ah, 'correlation').ravel()
        #eucs = -cdist(target_alpha, ah, 'euclidean').ravel()
        
        '''
        pca = PCA()
        pca.fit(np.concatenate((ah, target_alpha), axis=0))
    
        ah_transform = pca.transform(ah)
        ta_transform = pca.transform(target_alpha)
        topaxis = ah_transform.shape[1]#100
        #topaxis = np.max([int((np.cumsum(pca.explained_variance_ratio_)<0.9).sum()), 2])
        ah_transform = ah_transform[:,:topaxis]
        ta_transform = ta_transform[:,:topaxis]
    
        cors_a = -cdist(ta_transform, ah_transform, 'correlation').ravel()
        eucs = -cdist(ta_transform, ah_transform, 'euclidean').ravel()
        '''
    
        #rankperf_house.append([cors_a, eucs])
        rankperf_house.append([cors_a])
            
        sc.settings.verbosity = 3
        
        result = pd.DataFrame({'Perturbation agents':totalpert[liveidx][np.argsort(cors_a)], 'score': np.sort(cors_a)}).sort_values('score', ascending=False)
    
        try:
            if save_path is not None:
                with open(save_path+'.p','wb') as f:
                    pickle.dump({'result':result}, f)
        except Exception as e:
            print('Error.', e)
            
        return result
    
    def calc_bing(self, mat):
        """
        Calculate bing (Best Inferred Genes from LINCS)

        Parameters
        ----------
        mat : np.array
            Gene expression values (n_samples, n_genes).

        Returns
        -------
        mat_and_bing : np.array
            Gene expression values of landmark genes and bing (n_samples, (n_genes+n_genes_expanded)).
        genesymbol : list
            Gene symbols.

        """
        
        assert(mat.shape[1] == self.Ws.shape[1])
        bing_genes = np.dot(mat, self.Ws.T.values) + self.b.values.reshape([1,-1])
        
        mat_and_bing = np.concatenate((mat, bing_genes), axis=1)
        genesymbol = self.landmark_genes['pr_gene_symbol'].values
        genesymbol = np.concatenate((genesymbol, self.Ws.index))
        
        return mat_and_bing, genesymbol
        
    def gene_expression_pert_simulation(self, initial_state, perturbation):
        """
        Simulating gene expression changes against given perturbation.

        Parameters
        ----------
        initial_state : np.array
            shape n_samples x n_genes containing gene expression values.
        perturbation : str
            Name of perturbation.

        Raises
        ------
        ValueError
            Raise error when the "perturbation" is not included in our data.

        Returns
        -------
        whole_gen_ori : list of np.array
            Simulated gene expression values [101, (n_samples, (n_genes+n_genes_expanded))].
        genesymbol : list
            Gene symbols.
        ret : pd.DataFrame
            Summarized simulated gene expression values (101, (n_genes+n_genes_expanded)). First row: initial state, Last row: perturbed state.

        """
        
        sc.settings.verbosity = 0

        train_data_use = self.train_data
        train_data_info_use = self.train_data_info
        
        totalccl = np.unique(train_data_info_use[:,0])
        totalpert = np.unique(train_data_info_use[:,1])
        totalpert = totalpert[totalpert!='control']
        
        if perturbation not in totalpert:
            raise ValueError(perturbation+' is not included in our data.')
        
        
        Aalpha = initial_state
        
        p = perturbation
        foralpha, foralphacontrol, aapcc = [], [], []
        for ccl in totalccl:
            if np.where((train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]==p))[0].size > 0:
                # print(ccl)
                np.random.seed(0)
                fa = train_data_use[(train_data_info_use[:,0]==ccl) & (train_data_info_use[:,1]==p),:]
                if (fa.shape[0] > 1) & (fa.shape[0] < 100):
                    fa_aug = np.repeat(fa, 2, axis=0) + np.random.normal(0, fa.std(axis=0, keepdims=True), size=(fa.shape[0]*2, fa.shape[1]))
                    fa = np.concatenate((fa, fa_aug), axis=0)
                foralpha.append(fa)
                fac = self.train_data_ve_[self.train_data_ve_info_[:,0]==ccl,:]
                if fac.shape[0] > 1:
                    fac_aug = np.repeat(fac, 2, axis=0) + np.random.normal(0, fac.std(axis=0, keepdims=True), size=(fac.shape[0]*2, fac.shape[1]))
                    fac = np.concatenate((fac, fac_aug), axis=0)  
                foralphacontrol.append(fac)
        
        if len(foralpha) > 0:
            alpha = []
            for cc in range(len(foralpha)):
                feed_dict = {self.model.t_term5:foralpha[cc], self.model.t_term6:foralphacontrol[cc], self.model.is_training:False}
                cmapping, calpha = self.sess.run([self.model.t6_mapping, self.model.perteffect], feed_dict=feed_dict)
                alpha.append(calpha)
        
                aapcc.append(-np.mean(cdist(Aalpha, foralphacontrol[cc], metric='correlation').ravel()))
        
            e_x = np.exp(aapcc - np.max(aapcc))# ** 5
            pccweight = e_x / e_x.sum()
            alpha = np.average(alpha, axis=0, weights=pccweight)
        
        sc.settings.verbosity = 3
        
        
        use_alpha = alpha
        
        whole_gen = []
        for d in np.arange(0.00, 1.01, 0.01):
            feed_dict = {self.model.t_term5:Aalpha, self.model.pertalpha:use_alpha, self.delta:d, self.model.is_training:False}
            whole_gen.append(self.sess.run(self.model.generated_interpolation, feed_dict=feed_dict))
            
        whole_gen_ori = whole_gen.copy()
        whole_gen = [_.mean(axis=0, keepdims=True) for _ in whole_gen]
        summary_gen = np.concatenate(whole_gen, axis=0)
                
        # Expand genes
        summary_gen, genesymbol = self.calc_bing(summary_gen)
        
        ret = pd.DataFrame(data=summary_gen, columns=genesymbol)
        return whole_gen_ori, genesymbol, ret
    
    def training(self, training_data=None, n_epoch=100, save_path='.', do_initialize=True):
        """
        Model training
        trained model can be accessed through self.model

        Parameters
        ----------
        training_data : pd.DataFrame, optional
            Training data. First two columns are cell and perturbation. Perturbation of control samples(==basal gene expression) should be "control". The default is None.
        n_epoch : int, optional
            Epoch. The default is 100.
        save_path : str, optional
            Save path for model checkpoint. The default is '.'.
        do_initialize : bool, optional
            Initialize model weights if True. The default is True.

        Returns
        -------
        None.

        """
        if training_data is not None:
            self.train_data_info_use = training_data.iloc[:,:infoidx].values
            self.train_data_use = training_data.iloc[:,infoidx:].values
        else:
            self.train_data_info_use = self.train_data_info
            self.train_data_use =self.train_data
        
        val_data = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p1.csv', index_col=None)
        val_data_info = val_data.iloc[:,:2].values
        val_data = val_data.iloc[:,2:].values
        val_data2 = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p2.csv', index_col=None)
        val_data2_info = val_data2.iloc[:,:2].values
        val_data2 = val_data2.iloc[:,2:].values
        val_data_ve = pd.read_csv('l1000_data/p1shp2xpr/validation_data_vehicle.csv', index_col=None)
        val_data_ve_info = val_data_ve.iloc[:,:2].values
        val_data_ve = val_data_ve.iloc[:,2:].values
        dmso = val_data_ve_info[:,1] == 'DMSO'
        val_data_ve = val_data_ve[dmso]
        val_data_ve_info = val_data_ve_info[dmso]
        val_data_ve_info[:,1] = 'control'

        val_data_info = np.concatenate((val_data_info, val_data2_info), axis=0)
        val_data = np.concatenate((val_data, val_data2), axis=0)
        filtercells = np.intersect1d(np.unique(val_data_info[:,0]), np.unique(val_data_ve_info[:,0]))
        idxfilt = np.full(val_data.shape[0], False)
        for vc in filtercells:
            tcc = val_data_info[:,0] == vc
            idxfilt = idxfilt | tcc
        self.val_data_info = np.concatenate((val_data_info[idxfilt], val_data_ve_info), axis=0)
        self.val_data = np.concatenate((val_data[idxfilt], val_data_ve), axis=0)
    
        np.random.seed(0)
        train_subsample = np.random.choice(self.train_data.shape[0], 4000, replace=False)
        train_controlsample = np.where(self.train_data_info[:,1]=='control')[0]
        train_subsample = np.unique(np.concatenate((train_subsample, train_controlsample)))
        tdi = self.train_data_info[train_subsample,:]
        td = self.train_data[train_subsample,:]
    
        val_subsample = np.random.choice(self.val_data.shape[0], 6000, replace=False)
        val_controlsample = np.where(self.val_data_info[:,1]=='control')[0]
        val_subsample = np.unique(np.concatenate((val_subsample, val_controlsample)))
        val_data_info = self.val_data_info[val_subsample,:]
        val_data = self.val_data[val_subsample,:]
    
        #train_dataset = Batch_maker(self.train_data_use, self.train_data_info_use, triweight=True)
        train_dataset = tdd = Batch_maker(td, tdi)
        val_dataset = Batch_maker(val_data, val_data_info)
    
        # training params
        batch_size = 32 
        val_freq= 50
        save_freq= 50

        if do_initialize:
            self.sess.run(tf.compat.v1.global_variables_initializer())
    
        summary_writer = tf.compat.v1.summary.FileWriter(save_path,self.sess.graph)
    
        for epoch in range(n_epoch):
            st = time()
            d_loss, g_loss, recon_loss, triple_loss, delta_loss, pl_lengths, pl_penalty, pact = run_train(self.sess, self.model, train_dataset, batch_size, self.genesym, self.pmodel)
    
            summ = tf.compat.v1.Summary()
            summ.value.add(tag='train/train_D_loss', simple_value=d_loss)
            summ.value.add(tag='train/train_G_loss', simple_value=g_loss)
            summ.value.add(tag='train/train_recon_loss', simple_value=recon_loss)
            summ.value.add(tag='train/train_triple_loss', simple_value=triple_loss)
            summ.value.add(tag='train/train_delta_loss', simple_value=delta_loss)
            # summ.value.add(tag='train/train_pl_lengths', simple_value=pl_lengths)
            # summ.value.add(tag='train/train_pl_penalty', simple_value=pl_penalty)
            summ.value.add(tag='train/train_pact_cor', simple_value=pact)
            summary_writer.add_summary(summ, epoch)
    
            if epoch % val_freq == 0:
                d_loss_trainsub, g_loss_train_sub, recon_loss_train_sub, triple_loss_train_sub, delta_loss_train_sub, pl_lengths_train, pl_penalty_train, errors_d_train, errors_d_z_train, pact_train = run_val(self.sess, self.model, tdd, batch_size, None, self.genesym, self.pmodel)
                d_loss_val, g_loss_val, recon_loss_val, triple_loss_val, delta_loss_val, pl_lengths_val, pl_penalty_val, errors_d, errors_d_z, pact_val = run_val(self.sess, self.model, val_dataset, batch_size, None, self.genesym, self.pmodel)
    
                summ = tf.compat.v1.Summary()
                summ.value.add(tag='train_sub/train_sub_D_loss', simple_value=d_loss_trainsub)
                summ.value.add(tag='train_sub/train_sub_G_loss', simple_value=g_loss_train_sub)
                summ.value.add(tag='train_sub/train_sub_recon_loss', simple_value=recon_loss_train_sub)
                summ.value.add(tag='train_sub/train_sub_triple_loss', simple_value=triple_loss_train_sub)
                summ.value.add(tag='train_sub/train_sub_delta_loss', simple_value=delta_loss_train_sub)
                # summ.value.add(tag='train_sub/train_sub_pl_lengths', simple_value=pl_lengths_train)
                # summ.value.add(tag='train_sub/train_sub_pl_penalty', simple_value=pl_penalty_train)
                # summ.value.add(tag='train_sub/train_sub_RFE', simple_value=errors_d_train)
                # summ.value.add(tag='train_sub/train_sub_RFE_z', simple_value=errors_d_z_train)
                summ.value.add(tag='train_sub/train_sub_pact_cor', simple_value=pact_train)
                summ.value.add(tag='val/val_D_loss', simple_value=d_loss_val)
                summ.value.add(tag='val/val_G_loss', simple_value=g_loss_val)
                summ.value.add(tag='val/val_recon_loss', simple_value=recon_loss_val)
                summ.value.add(tag='val/val_triple_loss', simple_value=triple_loss_val)
                summ.value.add(tag='val/val_delta_loss', simple_value=delta_loss_val)
                # summ.value.add(tag='val/val_pl_lengths', simple_value=pl_lengths_val)
                # summ.value.add(tag='val/val_pl_penalty', simple_value=pl_penalty_val)
                # summ.value.add(tag='val/val_RFE', simple_value=errors_d)
                # summ.value.add(tag='val/val_RFE_z', simple_value=errors_d_z)
                summ.value.add(tag='val/val_pact_cor', simple_value=pact_val)
                summary_writer.add_summary(summ, epoch)
                
                print('epoch',epoch,'/',n_epoch,' - train sub D loss:', d_loss_trainsub,
                      ', train sub G loss:', g_loss_train_sub, ', train sub errors_d', errors_d_train, ', train sub errors_d_z', errors_d_z_train, ', train sub pact', pact_train)
                print('epoch',epoch,'/',n_epoch,' - val D loss:', d_loss_val, ', val G loss:', g_loss_val, ', val errors_d', errors_d, ', val errors_d_z', errors_d_z, ', val pact', pact_val)
    
            print('epoch',epoch,'/',n_epoch,' - D loss:', d_loss, ', G loss:', g_loss, 'Elapsed..', time()-st)
            sys.stdout.flush()
            if epoch % save_freq == 0:
                self.saver.save(self.sess, save_path+'/model-'+str(epoch)+'.ckpt')
                
        self.saver.save(self.sess, save_path+'/model-final.ckpt')