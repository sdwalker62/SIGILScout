python3 \
-m torch.distributed.launch \
--nproc_per_node 4 \
yolov5/train.py \
--data yolov5/Staplers-3/data.yaml \
--weights yolov5n.pt \
--hyp yolov5/hyp_evolve.yaml \
--cfg yolov5/models/yolov5n.yaml \
--batch-size 32 \
--epochs 3000 \
--entity peerteam \
--project results/DOTA \
--device 0,1,2,3
