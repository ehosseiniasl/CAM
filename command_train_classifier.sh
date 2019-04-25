
NAME=voc_fc_resnet50
ARCH=fc_resnet50
BATCH=$3
LR=0.001

CUDA_VISIBLE_DEVICES=$1 python pascal_prm_classification.py --gpu $1 -ims $2 --lr $LR -b $BATCH --arch $ARCH --name $NAME --pretrained
