# With best hyperparameters

# Retrieval: MF + Gowalla
python train.py model=MF hash_type=full dataset=Gowalla dataset.lr=5e-3 dataset.wd=1e-8
python train.py model=MF hash_type=graph dataset=Gowalla dataset.lr=1e-2 dataset.wd=1e-6
python train.py model=MF hash_type=frequency dataset=Gowalla dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=random dataset=Gowalla dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=lsh-structure dataset=Gowalla dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double dataset=Gowalla dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double_frequency dataset=Gowalla dataset.lr=1e-3 dataset.wd=1e-6



# Retrieval: MF + Yelp2018
python train.py model=MF hash_type=full dataset=Yelp2018 dataset.lr=1e-2 dataset.wd=1e-6
python train.py model=MF hash_type=graph dataset=Yelp2018 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=frequency dataset=Yelp2018 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=random dataset=Yelp2018 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=lsh-structure dataset=Yelp2018 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double dataset=Yelp2018 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double_frequency dataset=Yelp2018 dataset.lr=1e-3 dataset.wd=1e-6


# Retrieval: MF + AmazonBook


python train.py model=MF hash_type=full dataset=AmazonBook dataset.lr=1e-2 dataset.wd=1e-8
python train.py model=MF hash_type=graph dataset=AmazonBook dataset.lr=1e-2 dataset.wd=1e-6
python train.py model=MF hash_type=frequency dataset=AmazonBook dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=random dataset=AmazonBook dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=MF hash_type=lsh-structure dataset=AmazonBook dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double dataset=AmazonBook dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=MF hash_type=double_frequency dataset=AmazonBook dataset.lr=1e-3 dataset.wd=1e-8



# Retrieval: NeuMF + Gowalla

python train.py model=NeuMF hash_type=full dataset=Gowalla dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-8
python train.py model=NeuMF hash_type=graph dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=frequency dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=random dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=lsh-structure dataset=Gowalla dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-4
python train.py model=NeuMF hash_type=double dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=double_frequency dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4





# Retrieval: NeuMF + Yelp2018

python train.py model=NeuMF hash_type=full dataset=Yelp2018 dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-8
python train.py model=NeuMF hash_type=graph dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=frequency dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=random dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=lsh-structure dataset=Yelp2018 dataset.bs=1024  dataset.lr=1e-2 dataset.wd=1e-4
python train.py model=NeuMF hash_type=double dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=NeuMF hash_type=double_frequency dataset=Yelp2018 dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-4



# Retrieval: NeuMF + AmazonBook

python train.py model=NeuMF hash_type=full dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=NeuMF hash_type=graph dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=NeuMF hash_type=frequency dataset=AmazonBook dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=random dataset=AmazonBook dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=lsh-structure dataset=AmazonBook dataset.bs=1024  dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=double dataset=AmazonBook dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=NeuMF hash_type=double_frequency dataset=AmazonBook dataset.bs=1024  dataset.lr=5e-3 dataset.wd=1e-6



# Retrieval: LightGCN + Gowalla

python train.py model=LightGCN hash_type=full dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=graph dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=frequency dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=random dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=lsh-structure dataset=Gowalla dataset.bs=1024  dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=double dataset=Gowalla dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=double_frequency dataset=Gowalla dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-6


# Retrieval: LightGCN + Yelp2018

python train.py model=LightGCN hash_type=full dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=graph dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=frequency dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=random dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-
python train.py model=LightGCN hash_type=lsh-structure dataset=Yelp2018 dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=double dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=LightGCN hash_type=double_frequency dataset=Yelp2018 dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-6



# Retrieval: LightGCN + AmazonBook

python train.py model=LightGCN hash_type=full dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=graph dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=frequency dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=random dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=lsh-structure dataset=AmazonBook dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=double dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=LightGCN hash_type=double_frequency dataset=AmazonBook dataset.bs=1024  dataset.lr=1e-3 dataset.wd=1e-6


# Retrieval: MFDAU + Gowalla: MF backbone with DirectAU loss

python train.py model=MFDAU hash_type=full dataset=Gowalla dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=graph dataset=Gowalla dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=frequency dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=random dataset=Gowalla dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-6 dataset.loss=DAU
python train.py model=MFDAU hash_type=lsh-structure dataset=Gowalla dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6 dataset.loss=DAU
python train.py model=MFDAU hash_type=double dataset=Gowalla dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-4 dataset.loss=DAU
python train.py model=MFDAU hash_type=double_frequency dataset=Gowalla dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6 dataset.loss=DAU

# Retrieval: MFDAU + Yelp2018

python train.py model=MFDAU hash_type=full dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=graph dataset=Yelp2018 dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6 dataset.loss=DAU
python train.py model=MFDAU hash_type=frequency dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=random dataset=Yelp2018 dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-6 dataset.loss=DAU
python train.py model=MFDAU hash_type=lsh-structure dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-2 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=double dataset=Yelp2018 dataset.bs=1024 dataset.lr=5e-3 dataset.wd=1e-4 dataset.loss=DAU
python train.py model=MFDAU hash_type=double_frequency dataset=Yelp2018 dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4 dataset.loss=DAU

# Retrieval: MFDAU + AmazonBook

python train.py model=MFDAU hash_type=full dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=graph dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=frequency dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=random dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6 dataset.loss=DAU
python train.py model=MFDAU hash_type=lsh-structure dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-8 dataset.loss=DAU
python train.py model=MFDAU hash_type=double dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-4 dataset.loss=DAU
python train.py model=MFDAU hash_type=double_frequency dataset=AmazonBook dataset.bs=1024 dataset.lr=1e-3 dataset.wd=1e-6 dataset.loss=DAU




# Ranking: DLRM + Frappe

python train.py model=DLRMs hash_type=full dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=graph dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=frequency dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=random dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=lsh-structure dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-8
python train.py model=DLRMs hash_type=double dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=double_frequency dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRMs hash_type=double_graph dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4





# Ranking: DCNv2 + Frappe 

python train.py model=DCNv2s hash_type=full dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=DCNv2s hash_type=graph dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2s hash_type=frequency dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2s hash_type=random dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2s hash_type=lsh-structure dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-6
python train.py model=DCNv2s hash_type=double dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2s hash_type=double_frequency dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2s hash_type=double_graph dataset=Frappe dataset.lr=5e-3 dataset.wd=1e-4


# Ranking: WideDeep + Frappe 

python train.py model=WideDeeps hash_type=full dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=graph dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=frequency dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=random dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=lsh-structure dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=double dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-6
python train.py model=WideDeeps hash_type=double_frequency dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeeps hash_type=double_graph dataset=Frappe dataset.lr=1e-3 dataset.wd=1e-4


# Ranking: DLRM + MovieLens1M

python train.py model=DLRM hash_type=full dataset=MovieLens1M-ranking dataset.lr=1e-2 dataset.wd=1e-6
python train.py model=DLRM hash_type=graph dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=frequency dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=random dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=lsh-structure dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=double dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=double_frequency dataset=MovieLens1M-ranking dataset.lr=1e-2 dataset.wd=1e-8
python train.py model=DLRM hash_type=double_graph dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DLRM hash_type=lsh dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-8



# Ranking: DCNv2 + MovieLens1M 

python train.py model=DCNv2 hash_type=full dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=graph dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=frequency dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=random dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=lsh-structure dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=double dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=double_frequency dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=double_graph dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=DCNv2 hash_type=lsh dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4

# Ranking: WideDeep + MovieLens1M 

python train.py model=WideDeep hash_type=full dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=graph dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=frequency dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=random dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=lsh-structure dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=double dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=double_frequency dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=double_graph dataset=MovieLens1M-ranking dataset.lr=5e-3 dataset.wd=1e-4
python train.py model=WideDeep hash_type=lsh dataset=MovieLens1M-ranking dataset.lr=1e-3 dataset.wd=1e-4
