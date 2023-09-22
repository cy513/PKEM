## Environment
    python 3.8
    pytorch 2.0.0
    dgl 1.1.0

## Run the experiment

Get the attributes of entities for static graph.

    python ent2attr.py --dataset DATA_NAME

Train the model.

    python train.py --dataset DATA_NAME -lr 0.001 --n-epoch 60 --hidden-dim 200 -gpu 1 --batch-size 1024 --joint_model 1
    
Test the model.

    python test.py --dataset DATA_NAME


